import json
import base64
import torch
import numpy as np
import pandas as pd
import os
import csv
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from openai import OpenAI


client = OpenAI(api_key="")

class QwenVLDescriber:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def describe_image(self, problem, image_path: str, feedback: str = None) -> str:
        image = Image.open(image_path).convert("RGB")
        image_example1 = Image.open("images/image_example1.png").convert("RGB")
        image_example2 = Image.open("images/image_example2.png").convert("RGB")
        image_example3 = Image.open("images/image_example3.png").convert("RGB")
        example_parse1 = f"""
                You are a geometry diagram parser.
                Convert the geometry diagram into symbolic clauses.
                Example Problem 1 (Image 1):

                Question 1:
                As shown in the figure, in \\odot X, A B = 30, C D = 30, and m \\widehat C Z = 40. Find m \\widehat A B

                Description:

                circle X
                A, B, C, D, Y, Z lie on circle X

                chord AB
                chord CD

                line AB
                line CD
                line XM
                line XN
                line XMY
                line XNZ

                M lies on AB
                N lies on CD

                M is the intersection of XM and AB
                N is the intersection of XN and CD

                XM perpendicular AB at M
                XN perpendicular CD at N

                X, M, Y are collinear
                X, N, Z are collinear

                XA = XB = XC = XD = XY = XZ   // all are radii of circle X

                AB = 30
                CD = 30

                arc CZ = 40

                ---
                """
        example_parse2 = f"""
                You are a geometry diagram parser.
                Convert the geometry diagram into symbolic clauses.
                Example Problem 2 (Image 2):

                Question 2:
                Find y.

                Description:

                Triangle ABC lies on A B C
                line AB
                line BC
                line AC

                ∠A = 90°.
                ∠B = 30°.

                BC = 18.

                AB = y.
                AC = x.
                ---
                """
        example_parse3 = f"""
                You are a geometry diagram parser.
                Convert the geometry diagram into symbolic clauses.
                Example Problem 3 (Image 3):

                Question 2:
                Find y.

                Description:

                Triangle ABC lies on A B C
                Triangle ADC lies on A D C
                Triangle ABC and Triangle ADC share base AC
                line AB
                line BC
                line CD
                line DA
                line AC
                line DB

                E is the intersection of AC and DB
                AC perpendicular DB at E

                AE = EC = EB

                ∠AEB = 90°.
                ∠BEC = 30°.
                ∠ADB = x°.

                AB = 8.
                BC = 8.
                AD = 10.
                DC = 10.

                ---
                """
        prompt_text = f"""
                Now parse the following problem with the given figure (Image 4) using exact the same format as the example problem.

                Don't do reasoning to solve the problem

                Important rules:



                1. Identify all important geometric points in the diagram, If the diagram already provides point names, use those names and don't need to specially introduce them.

                2. Extract only facts that are explicitly visible in the diagram or explicitly stated in the question.
                3. Then add standard properties that are directly implied by named geometric shapes.
                  For example, if the figure is a parallelogram ABCD, always include:
                  - AB ∥ CD
                  - AD ∥ BC
                  - AB = CD
                  - AD = BC
                4. Do NOT invent unsupported angle relations.
                5. Do NOT say a segment bisects an angle unless the problem explicitly states it or there is an unambiguous angle-bisector mark.
                6. If a segment bisects an angle, points out which two angles are equivalent
                  For example, it CD bisects ACB, you may infer:
                  angle ACD = angle BCD
                7. If a point lies on a segment, state it explicitly.
                8. Output one clause per line.
                9. Do not explain your reasoning.
                10. Use the symbol "=" for equality.


                Question:
                {question}
                Description:

                """

        messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Example image1:"},
                {"type": "image"},
                {"type": "text", "text": example_parse1},
                {"type": "text", "text": "Example image2:"},
                {"type": "image"},
                {"type": "text", "text": example_parse2},
                {"type": "text", "text": "Example image3:"},
                {"type": "image"},
                {"type": "text", "text": example_parse3},
                {"type": "image"},
                {"type": "text", "text": prompt_text},

            ]
        }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                # do_sample=False,  # 跑批建议关采样，稳定
                do_sample=True,
                temperature=0.7,
                top_p=0.9,

            )

        generated_ids = out[:, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return text

def qwen_generate(problem, image_path=None, feedback=None):
    model = QwenVLDescriber()
    response = model.describe_image(problem, image_path=image_path)
    return response

def judge_with_gpt5mini(problem, image_path, candidate):
    with open(image_path, "rb") as f:
      image_base64 = base64.b64encode(f.read()).decode("utf-8")
    image_data_url = f"data:image/png;base64,{image_base64}"
    prompt = f"""
            You are a strict evaluator for visual geometry description.
            Use BOTH the problem text and the provided image to evaluate the quality of description quality.

            You are given:
            1. the geometry problem text,
            2. the diagram,
            3. a candidate description of the diagram.

            Evaluate the candidate description using the rubric below.(Total 0–4.0 points)

            Rubric:
            1. Existence: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly
            confidently incorrect. Does the description correctly identify points, segments, angles and geometric objects
            that actually appear in the image?
            2. Attribute Accuracy: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means
            mostly confidently incorrect. Are the described facts (relationship, size, etc.) are correct with respect to the image and problem text?
            3. Completeness: Assign scores from 0 to 1.0, where 1.0 means mostly confidently correct and 0 means mostly
            confidently incorrect. Does the description include all key objects and necessary details relevant to the
            image?
            4. Overall: Sum up the score of three aspects to get the overall score.

            Problem:
            {problem}

            Candidate:
            {candidate}

            Return valid JSON only:
            {{
            "Existence": 0.0,
            "Attribute Accuracy": 0.0,
            "Completeness": 0.0,
            "Overall": 0.0,
            "feedback": "..."
            }}
            """
    resp = client.responses.create(
        model="gpt-5-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": image_data_url
                }
            ]
        }]
    )
    return json.loads(resp.output_text)

def search_best_answer(problem, image_path, num_samples=1, scoring_rounds=1):
    rows = []
    for i in range(num_samples):
        candidate = qwen_generate(problem, image_path)
        # candidate = """
        # Description:
        # parallelogram WXZY
        # line WX
        # line XY
        # line YZ
        # line ZX
        # line WY
        # line XZ

        # m ∠WXY = 82°
        # m ∠XYZ = 33°



        # ---
        # """
        print(candidate)

        Existence_scores = []
        Attribute_Accuracy_scores = []
        Completeness_scores = []
        Overall_scores = []

        for _ in range(scoring_rounds):
            cur_result = judge_with_gpt5mini(problem, image_path, candidate)
            Existence_scores.append(cur_result["Existence"])
            Attribute_Accuracy_scores.append(cur_result["Attribute Accuracy"])
            Completeness_scores.append(cur_result["Completeness"])
            Overall_scores.append(cur_result["Overall"])

        Mean_Existence_score = np.mean(Existence_scores)
        Mean_Attribute_Accuracy_score = np.mean(Attribute_Accuracy_scores)
        Mean_Completeness_score = np.mean(Completeness_scores)
        Mean_Overall_score = np.mean(Overall_scores)

        # if result["overall"] > best_score:
        #     best_score = result["overall"]
        #     best = {
        #         "answer": candidate,
        #         "judge": result
        #     }
        rows.append({
            "problem": problem,
            "try": i,
            "Existence_score": Mean_Existence_score,
            "Attribute_Accuracy_score": Mean_Attribute_Accuracy_score,
            "Completeness_score": Mean_Completeness_score,
            "Overall_score": Mean_Overall_score
        })
    return rows




if __name__ == "__main__":
    base_dir = "./geo3k/train"
    csv_path = "results.csv"
    all_rows = []
    for qid in os.listdir(base_dir):  # loop through problems, load problem text and images
      q_path = os.path.join(base_dir, qid)
      if not os.path.isdir(q_path):
        continue
      image_path = os.path.join(q_path, "img_diagram_point.png")
      json_path = os.path.join(q_path, "data.json")
      with open(json_path, "r") as f:
        problem = json.load(f)["problem_text"]
      # problem = "W X Y Z is a parallelogram. Find m \\angle Y Z W."
      # image_path = "./images/image_q49_points.png"

      rows = search_best_answer(problem, image_path)

      all_rows.extend(rows)

    headers = [
        "problem",
        "sample_id",
        "Existence",
        "Attribute_Accuracye",
        "Completeness",
        "Overall"
    ]
    with open("results.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(headers)      # 👈 header row
      writer.writerows(all_rows)    # 👈 your data