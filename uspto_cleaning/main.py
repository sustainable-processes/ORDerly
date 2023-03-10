import datetime
import argparse

import uspto_cleaning.USPTO_cleaning


if __name__ == "__main__":
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_data_file_name', type=str, default='cleaned_USPTO')
    parser.add_argument('--consistent_yield', type=bool, default=True)
    parser.add_argument('--num_reactant', type=int, default=5)
    parser.add_argument('--num_product', type=int, default=5)
    parser.add_argument('--num_solv', type=int, default=2)
    parser.add_argument('--num_agent', type=int, default=3)
    parser.add_argument('--num_cat', type=int, default=0)
    parser.add_argument('--num_reag', type=int, default=0)
    parser.add_argument('--min_frequency_of_occurance_primary', type=int, default=15)
    parser.add_argument('--min_frequency_of_occurance_secondary', type=int, default=15)
    parser.add_argument('--include_other_category', type=bool, default=True)
    parser.add_argument('--save_with_label_called_other', type=int, default=3) #save the reaction: label the rare molecule with 'other' rather than removing it
    

    args = parser.parse_args()

    # Access the arguments as attributes of the args object
    clean_data_file_name = args.clean_data_file_name
    consistent_yield = args.consistent_yield
    num_reactant = args.num_reactant
    num_product = args.num_product
    num_solv = args.num_solv
    num_agent = args.num_agent
    num_cat = args.num_cat
    num_reag = args.num_reag
    min_frequency_of_occurance_primary = args.min_frequency_of_occurance_primary
    min_frequency_of_occurance_secondary = args.min_frequency_of_occurance_secondary
    include_other_category = args.include_other_category
    save_with_label_called_other = args.save_with_label_called_other
        
    assert num_agent == 0 or num_cat == 0 and num_reag == 0, "Invalid input: If merge_conditions=True in USPTO_extraction, then num_cat and num_reag must be 0. If merge_conditions=False, then num_agent must be 0."
    
    assert min_frequency_of_occurance_primary > save_with_label_called_other and min_frequency_of_occurance_secondary > save_with_label_called_other, "min_frequency_of_occurance_primary and min_frequency_of_occurance_secondary must be greater than save_with_label_called_other. Anything between save_with_label_called_other and min_frequency_of_occurance_primary/secondary will be set to 'other' if include_other_category=True."

    instance = uspto_cleaning.USPTO_cleaning.USPTO_cleaning(clean_data_file_name, consistent_yield, num_reactant, num_product, num_solv, num_agent, num_cat, num_reag, min_frequency_of_occurance_primary, min_frequency_of_occurance_secondary, include_other_category, save_with_label_called_other)
    
    instance.main()
        
    end_time = datetime.now()

    print('Duration: {}'.format(end_time - start_time))

