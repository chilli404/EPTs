import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from fogen.model import GPT, ModelConfig


def _is_empty_cache(cache) -> bool:
    """True for None or a freshly-constructed, not-yet-populated DynamicCache."""
    if cache is None:
        return True
    if isinstance(cache, DynamicCache):
        return not cache.layers or cache.layers[0].keys is None
    return False


def _to_legacy_cache(cache):
    """Convert a DynamicCache (or None) to GPT.forward_cached's (k, v)-list format."""
    if _is_empty_cache(cache):
        return None
    if isinstance(cache, DynamicCache):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return cache


class FogenConfig(PretrainedConfig):
    model_type = "fogen"

    def __init__(
        self,
        vocab_size=8192,
        n_layer=4,
        d_model=256,
        n_head=2,
        ctx_len=2048,
        execution_mode="sequential",
        norm_type="layernorm",
        mlp_type="relu2",
        use_value_embeddings=True,
        logit_softcap=15.0,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer
        self.d_model = d_model
        self.hidden_size = d_model
        self.intermediate_size = 4 * d_model
        self.num_attention_heads = n_head
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.max_position_embeddings = ctx_len
        self.execution_mode = execution_mode
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.use_value_embeddings = use_value_embeddings
        self.logit_softcap = logit_softcap
        self.architectures = ["FogenForCausalLM"]

    def model_config(self):
        return ModelConfig(
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            d_model=self.d_model,
            n_head=self.n_head,
            ctx_len=self.ctx_len,
            execution_mode=self.execution_mode,
            norm_type=self.norm_type,
            mlp_type=self.mlp_type,
            use_value_embeddings=self.use_value_embeddings,
            logit_softcap=self.logit_softcap,
        )


class FogenForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FogenConfig
    main_input_name = "input_ids"
    _tied_weights_keys = {}
    all_tied_weights_keys = {}
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(config.model_config())
        self.post_init()

    def _init_weights(self, module):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.model._rebuild_rope_cache()
        return model

    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, value):
        self.model.wte = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        use_cache=False,
        execution_mode=None,
        **kwargs,
    ):
        mode = execution_mode or self.config.execution_mode
        if use_cache or past_key_values is not None:
            logits, new_cache = self.model.forward_cached(
                input_ids, cache=_to_legacy_cache(past_key_values), mode=mode)
            cache = DynamicCache(ddp_cache_data=new_cache)
        else:
            logits = self.model(input_ids, mode=mode)
            cache = None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=cache,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, **kwargs
    ):
        if not _is_empty_cache(past_key_values):
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }


# Register with transformers' Auto* factories so AutoConfig/AutoModelForCausalLM
# can resolve config.json's "model_type": "fogen" once this module is imported.
AutoConfig.register("fogen", FogenConfig)
AutoModelForCausalLM.register(FogenConfig, FogenForCausalLM)
