from typing import *
from glue3d.evaluators.base import MetaJudge


class TraditionalCaptionMetricEvaluator(MetaJudge):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        from rouge import Rouge

        self.sbert_model = SentenceTransformer("all-mpnet-base-v2")
        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model.eval()
        self.rouge = Rouge()

    def judge_answer(self, ground_truth: str, model_output: str) -> Mapping[str, Any]:
        import torch
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        from scipy.spatial.distance import cosine
        from sentence_transformers import util

        # create a SmoothingFunction object
        smoothing_function = SmoothingFunction().method1  # * used to deal with non-overlap n-gram

        # calculate BLEU-1 score with smoothing function
        bleu_1_score = sentence_bleu(
            [ground_truth.split()], model_output.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function
        )

        # calculate BLEU-2, BLEU-3, and BLEU-4 scores
        bleu_2_score = sentence_bleu(
            [ground_truth.split()],
            model_output.split(),
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing_function,
        )
        bleu_3_score = sentence_bleu(
            [ground_truth.split()],
            model_output.split(),
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smoothing_function,
        )
        bleu_4_score = sentence_bleu(
            [ground_truth.split()],
            model_output.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_function,
        )

        rouge_scores_l = self.rouge.get_scores(model_output, ground_truth)[0]["rouge-l"]

        # calculate METEOR score
        meteor_scores = meteor_score([ground_truth.split()], model_output.split())

        # Calculate SBERT similarity
        embeddings = self.sbert_model.encode([ground_truth, model_output])
        sbert_similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()

        # calculate SimCSE similarity
        # Tokenize input texts
        inputs = self.simcse_tokenizer([ground_truth, model_output], padding=True, truncation=True, return_tensors="pt")

        # Get the embeddings
        with torch.no_grad():
            embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        # Calculate cosine similarity
        simcse_similarity = 1 - cosine(
            embeddings[0], embeddings[1]
        )  # * consine actually calculates consine distance, which is 1 - consine similarity

        return {
            "BLEU-1": bleu_1_score * 100,
            "BLEU-2": bleu_2_score * 100,
            "BLEU-3": bleu_3_score * 100,
            "BLEU-4": bleu_4_score * 100,
            "METEOR": meteor_scores * 100,
            "ROUGE-L": rouge_scores_l["f"] * 100,
            "SBERT_SCORE": sbert_similarity * 100,
            "SIMCSE_SCORE": float(simcse_similarity) * 100,
        }
