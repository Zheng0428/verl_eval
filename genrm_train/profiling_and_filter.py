import json
import argparse
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def profile_data(input_file):
    """Profile the processed data to analyze reward scores."""
    logging.info(f"Reading processed data from {input_file}")
    data = read_json(input_file)
    
    # Initialize counters
    genrm_scores = {"0.0": 0, "1.0": 0, "other": 0, "missing": 0}
    four_o_scores = {0.0: 0, 1.0: 0, "other": 0, "missing": 0}
    total_items = len(data)
    # Iterate through data items
    for item in data:
        extra_info = item.get("extra_info", {})
        
        # Count GenRM Reward Score
        genrm_score = extra_info.get("GenRM Reward Score")
        if genrm_score is None:
            genrm_scores["missing"] += 1
        elif genrm_score == "0.0" or genrm_score == 0.0:
            genrm_scores["0.0"] += 1
        elif genrm_score == "1.0" or genrm_score == 1.0:
            genrm_scores["1.0"] += 1
        else:
            genrm_scores["other"] += 1
            
        # Count 4o Reward Score
        four_o_score = extra_info.get("4o Reward Score")
        if four_o_score is None:
            four_o_scores["missing"] += 1
        elif four_o_score == 0.0 or four_o_score == "0.0":
            four_o_scores[0.0] += 1
        elif four_o_score == 1.0 or four_o_score == "1.0":
            four_o_scores[1.0] += 1
        else:
            four_o_scores["other"] += 1
    
    # Print results
    logging.info(f"Total items: {total_items}")
    
    logging.info("\nGenRM Reward Score:")
    logging.info(f"  0.0: {genrm_scores['0.0']} items ({genrm_scores['0.0']/total_items*100:.2f}%)")
    logging.info(f"  1.0: {genrm_scores['1.0']} items ({genrm_scores['1.0']/total_items*100:.2f}%)")
    logging.info(f"  Other values: {genrm_scores['other']} items ({genrm_scores['other']/total_items*100:.2f}%)")
    logging.info(f"  Missing: {genrm_scores['missing']} items ({genrm_scores['missing']/total_items*100:.2f}%)")
    
    logging.info("\n4o Reward Score:")
    logging.info(f"  0.0: {four_o_scores[0.0]} items ({four_o_scores[0.0]/total_items*100:.2f}%)")
    logging.info(f"  1.0: {four_o_scores[1.0]} items ({four_o_scores[1.0]/total_items*100:.2f}%)")
    logging.info(f"  Other values: {four_o_scores['other']} items ({four_o_scores['other']/total_items*100:.2f}%)")
    logging.info(f"  Missing: {four_o_scores['missing']} items ({four_o_scores['missing']/total_items*100:.2f}%)")
    
    # Calculate agreement between scores
    both_zero = 0
    both_one = 0
    disagree = 0
    comparable = 0
    four_o_1_genrm_0 = 0
    four_o_0_genrm_1 = 0
    data_class = {
        "both_zero": [],
        "both_one": [],
        "disagree": [],
        "four_o_1_genrm_0": [],
        "four_o_0_genrm_1": []
    }
    for item in data:
        extra_info = item.get("extra_info", {})
        genrm_score = extra_info.get("GenRM Reward Score")
        four_o_score = extra_info.get("4o Reward Score")
        
        # Only compare when both scores are present
        if genrm_score is not None and four_o_score is not None:
            comparable += 1
            
            # Convert to same type for comparison
            genrm_float = float(genrm_score)
            four_o_float = float(four_o_score) if isinstance(four_o_score, str) else four_o_score
            
            if genrm_float == 0.0 and four_o_float == 0.0:
                both_zero += 1
                data_class["both_zero"].append(item)
            elif genrm_float == 1.0 and four_o_float == 1.0:
                both_one += 1
                data_class["both_one"].append(item)
            else:
                disagree += 1
                data_class["disagree"].append(item)
            if four_o_float == 1.0 and genrm_float == 0.0:
                four_o_1_genrm_0 += 1
                data_class["four_o_1_genrm_0"].append(item)
            if four_o_float == 0.0 and genrm_float == 1.0:
                four_o_0_genrm_1 += 1
                data_class["four_o_0_genrm_1"].append(item)
                
                
    if comparable > 0:
        logging.info("\nScore Agreement:")
        logging.info(f"  Both 0.0: {both_zero} items ({both_zero/comparable*100:.2f}%)")
        logging.info(f"  Both 1.0: {both_one} items ({both_one/comparable*100:.2f}%)")
        logging.info(f"  Disagreement: {disagree} items ({disagree/comparable*100:.2f}%)")
        logging.info(f"  Total agreement: {(both_zero + both_one)/comparable*100:.2f}%")
        logging.info(f"  4o 1.0, GenRM 0.0: {four_o_1_genrm_0} items ({four_o_1_genrm_0/comparable*100:.2f}%)")
        logging.info(f"  4o 0.0, GenRM 1.0: {four_o_0_genrm_1} items ({four_o_0_genrm_1/comparable*100:.2f}%)")
        
    return data_class
        
        
def write_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def filter_data(data_class, output_file, validation_file=None):
    
    both_zero = 1000
    both_one = 5000
    four_o_1_genrm_0 = 5000
    four_o_0_genrm_1 = 5000
    data_list = []
    validation_data = []
    
    for key, value in data_class.items():
        if key == "both_zero":
            random.shuffle(value)
            # Save 50 examples for validation if validation_file is provided
            if validation_file:
                validation_data.extend(value[:25])
                data_list.extend(value[25:both_zero+25])
            else:
                data_list.extend(value[:both_zero])
        elif key == "both_one":
            random.shuffle(value)
            # Save 50 examples for validation if validation_file is provided
            if validation_file:
                validation_data.extend(value[:25])
                data_list.extend(value[25:both_one+25])
            else:
                data_list.extend(value[:both_one])
        elif key == "four_o_0_genrm_1":
            random.shuffle(value)
            # Save 50 examples for validation if validation_file is provided
            if validation_file:
                validation_data.extend(value[:25])
                data_list.extend(value[50:four_o_0_genrm_1+50])
            else:
                data_list.extend(value[:four_o_0_genrm_1])
        elif key == "four_o_1_genrm_0":
            random.shuffle(value)
            # Save 50 examples for validation if validation_file is provided
            if validation_file:
                validation_data.extend(value[:25])
                data_list.extend(value[50:four_o_1_genrm_0+50])
            else:
                data_list.extend(value[:four_o_1_genrm_0])
            
    write_json(data_list, output_file)
    logging.info(f"Filtered data saved to {output_file}, length: {len(data_list)}")
    
    # Save validation data if validation_file is provided
    if validation_file and validation_data:
        write_json(validation_data, validation_file)
        logging.info(f"Validation data saved to {validation_file}, length: {len(validation_data)}")
            
def main():
    parser = argparse.ArgumentParser(description="Profile processed data for reward scores")
    parser.add_argument("--input", type=str, required=True, help="Path to processed JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--validation", type=str, default=None, help="Path to validation set JSON file")
    args = parser.parse_args()
    
    data_class = profile_data(args.input)
    filter_data(data_class, args.output, args.validation)

if __name__ == "__main__":
    main()



