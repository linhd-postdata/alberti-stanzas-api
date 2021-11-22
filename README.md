# API for the ALBERTI model fine-tuned for stanza detection

## Validation Metrics

- Loss: 1.3520570993423462
- Accuracy: 0.6083916083916084
- Macro F1: 0.5420169617715481
- Micro F1: 0.6083916083916084
- Weighted F1: 0.5963328136975058
- Macro Precision: 0.5864033493660455
- Micro Precision: 0.6083916083916084
- Weighted Precision: 0.6364793882921277
- Macro Recall: 0.5545405576555766
- Micro Recall: 0.6083916083916084
- Weighted Recall: 0.6083916083916084


## Usage

Install requirements:

```
pip install requirements.txt
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("alberti-finetuned", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("alberti-finetuned", local_files_only=True)

sample_stanza = """Mientras por competir con tu cabello,
oro bruñido al sol relumbra en vano;
mientras con menosprecio en medio el llano
mira tu blanca frente el lilio bello;"""

inputs = tokenizer(sample_stanza, return_tensors="pt")

outputs = model(**inputs)

best = pt.argmax(outputs.logits, dim=-1).item()

print(model.config.id2label[best])
```
