import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "harsha28/lexgpt-legal-reasoning-lfqa"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("patent/LexGPT-6B")

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("<|context|> Stat. § 14-32.1(a), does not make the definition an essential element of the crime pursuant to N.C. Gen. Stat. § 14-32.1(e). Therefore, we reject Defendant’s argument that it is not sufficient for the indictment to “merely state that the victim was ‘handicapped.’ ” Furthermore, the indictment provided Defendant with enough information to prepare a defense for the offense of felony assault on a handicapped person. See Leonard, _ N.C. App. at _, 711 S.E.2d at 873 (rejecting the defendant’s argument that the indictment was not sufficient because the indictment tracked the relevant language of the statute, listed “the essential elements of the offense[,]” and provided the defendant “with enough information to prepare a defense”); State v. Crisp, 126 N.C. App. 30, 36, 483 S.E.2d 462, 466 (<HOLDING>), appeal dismissed and disc. review denied, 346 <|question|> Is it necessary for the definition of the crime to be stated in the indictment according to N.C. Gen. Stat. § 14-32.1(a)? <|issues|>", return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=1000)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))