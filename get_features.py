from _modules import MarianMAMLFeatures, describe_features
from transformers import MarianTokenizer, MarianMTModel

model_name = "Helsinki-NLP/opus-mt-en-es"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

text = "This is a test."
inputs = tokenizer(text, return_tensors="pt")
decoder_inputs = tokenizer("<pad>", return_tensors="pt")  # force decoder input

feature_extractor = MarianMAMLFeatures(model)
features = feature_extractor(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    decoder_input_ids=decoder_inputs["input_ids"]
)

print(describe_features(features))

