"""Schema-guided tokens, not manual value repair; numeric ranges still validated."""
import json
import functools
import torch
from transformers import StoppingCriteria
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from lmformatenforcer.tokenenforcer import TokenEnforcer,TokenEnforcerTokenizerData
from ldc_main_contract_v32 import reply_schema


def make_parser(feedback):
    config=CharacterLevelParserConfig();config.max_consecutive_whitespaces=1;config.force_json_field_order=True
    return JsonSchemaParser(reply_schema(feedback),config=config)


def decode_tokens(tokenizer,tokens):
    return tokenizer.decode(tokens,clean_up_tokenization_spaces=False).rstrip('\ufffd')


def prepare_tokenizer(tokenizer):
    # The release's optional HF adapter imports an old Transformers class path.
    # Use its public core API with the current tokenizer, without changing packages.
    zero=tokenizer.encode('0',add_special_tokens=False)[-1];regular=[]
    leading=tokenizer.decode([zero],clean_up_tokenization_spaces=False)
    special=set(tokenizer.all_special_ids)
    for index in range(len(tokenizer)):
        if index in special:continue
        standalone=tokenizer.decode([index],clean_up_tokenization_spaces=False)
        contextual=tokenizer.decode([zero,index],clean_up_tokenization_spaces=False)[len(leading):]
        regular.append((index,contextual,len(contextual)>len(standalone)))
    return TokenEnforcerTokenizerData(regular,functools.partial(decode_tokens,tokenizer),tokenizer.eos_token_id,False,len(tokenizer))


def build_constraint(tokenizer_data,feedback):
    enforcer=TokenEnforcer(tokenizer_data,make_parser(feedback))
    def allowed(batch_id,tokens):return enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens
    return allowed


class CompleteJSON(StoppingCriteria):
    def __init__(self,tokenizer,prompt_length):self.tokenizer=tokenizer;self.prompt_length=prompt_length
    def __call__(self,input_ids,scores,**kwargs):
        raw=self.tokenizer.decode(input_ids[0,self.prompt_length:],skip_special_tokens=True)
        try:
            value=json.loads(raw);complete=isinstance(value,dict)
        except (ValueError,TypeError):complete=False
        return torch.tensor([complete],dtype=torch.bool,device=input_ids.device)
