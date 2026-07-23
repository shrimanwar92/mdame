from constants import get_onnx_model
import numpy as np
import pandas as pd

tokenizer, ort_session = get_onnx_model()

class Utils:

    @staticmethod
    def read_data(file_path):
        file_path = str(file_path)
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)

        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def save_to_parquet(df, out_path):
        df.to_parquet(out_path, compression="snappy", index=False)
        print(f"💾 Process Complete. Gold file saved to {out_path}.\n")

    @staticmethod
    def execute_onnx_embedding(text) -> np.ndarray:
        single_input = isinstance(text, str)

        if single_input:
            text = [text]
        
        encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
        
        onnx_inputs = {
            "input_ids": encoded_input["input_ids"].astype(np.int64),
            "attention_mask": encoded_input["attention_mask"].astype(np.int64)
        }
        if "token_type_ids" in encoded_input:
            onnx_inputs["token_type_ids"] = encoded_input["token_type_ids"].astype(np.int64)

        model_outputs = ort_session.run(None, onnx_inputs)
        token_embeddings = model_outputs[0] 
        attention_mask = encoded_input["attention_mask"]

        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        embeddings = sum_embeddings / sum_mask
        if single_input:
            return embeddings[0].flatten()

        return embeddings