# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# try:
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")
import json


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, is_debug=False) -> bool:
    if is_debug:
        print(f">>> {model_output=}, {ground_truth=}")
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = -1.0
    acc = False
    predictions_str = ""

    if not (ground_truth.startswith("\\boxed{")):
        # Wrap the ground truth in \boxed{} format for verification
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    else:
        ground_truth_boxed = ground_truth
    try:
        ret_score_verify, str_preds = verify_func([ground_truth_boxed], [model_output])
        predictions = {"predictions": str_preds[1]}
        predictions_str = json.dumps(predictions, ensure_ascii=False)
        if ret_score_verify > 0:
            acc = True
            ret_score = 1.0
        
    except Exception:
        predictions_str = "FailedVerification"
        pass
    except TimeoutException:
        ret_score = timeout_score
        predictions_str = "TimeoutVerification"

    ret = {
        "score": float(ret_score),
        'acc': acc,
        'pred': predictions_str
    }
    return ret


if __name__ == "__main__":
    print("Expected output: False False True True True True True")
    print(compute_score("\\boxed{123}", "456", is_debug=True))
    print(compute_score("\boxed{123}", "456", is_debug=True))
    print(compute_score("\\boxed{123}", "123", is_debug=True))
    print(compute_score("aaa\boxed{123}", "123", is_debug=True))  # \x08oxed{123} cannot be extracted successfully
    print(compute_score("aaa\\boxed{123}", "123", is_debug=True))
    print(compute_score("aaa\\\boxed{123}", "123", is_debug=True))  # # \x08oxed{123} cannot be extracted successfully
    print(compute_score("aaa\\\\boxed{123}", "123", is_debug=True))
