TRIPLES = {'t0_llava_gemma_vqa': {'base': 'llava_onevision_7b', 'verifier': 'gemma3_12b', 'set': 'vqav2_ood', 'base_backbone': 'Qwen2', 'verifier_backbone': 'Gemma3', 'independence': 'full: different family, backbone, tokenizer, vision stack', 'provenance': 'prior transfer study', 'items': 'data/snowglobe/frontier_lambda/pilot_eval_items.json', 'base_images_dir': 'data/snowglobe/verifier_escape/images'}, 't1_llava_internvl_vqa': {'base': 'llava_onevision_7b', 'verifier': 'internvl3_5_2b', 'set': 'vqav2_ood', 'base_backbone': 'Qwen2', 'verifier_backbone': 'Qwen3', 'independence': 'partial: Qwen3 backbone distinct from Qwen2 (different model, size, vision stack) but SHARED Qwen tokenizer lineage. Weakest independence; flagged.', 'items': 'data/snowglobe/frontier_lambda/pilot_eval_items.json', 'base_images_dir': 'data/snowglobe/verifier_escape/images'}, 't2_gemma_llava_gqa': {'base': 'gemma3_12b', 'verifier': 'llava_onevision_7b', 'set': 'gqa_val_eval', 'base_backbone': 'Gemma3', 'verifier_backbone': 'Qwen2', 'independence': 'full: roles swapped from the prior transfer study, different family and tokenizer', 'items': 'data/snowglobe/verifier_escape/t2_gemma_gqa_items.json', 'base_images_dir': 'data/snowglobe/verifier_escape/images'}, 't3_molmo_gemma_pope': {'base': 'molmo2_4b', 'verifier': 'gemma3_12b', 'set': 'pope_ood', 'base_backbone': 'Molmo2', 'verifier_backbone': 'Gemma3', 'independence': 'full: Molmo2 base, Gemma3 verifier, different family and tokenizer; binary POPE cell, hardest transfer case', 'items': 'data/snowglobe/verifier_escape/t3_molmo_pope_items.json', 'base_images_dir': 'data/snowglobe/verifier_escape/images'}}

def _merge_matrix():
    import json as _json
    from pathlib import Path as _Path
    p = _Path(__file__).resolve().parents[2] / 'data' / 'snowglobe' / 'verifier_escape' / 'matrix_triples.json'
    if p.exists():
        for k, v in _json.load(open(p)).items():
            if k not in TRIPLES:
                TRIPLES[k] = v
_merge_matrix()
