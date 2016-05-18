
function ret = get_training_set(data_filename)

    complete_dataset = load('-ascii', data_filename);
    dataset_sorted = sortrows(complete_dataset);

    patterns_count = rows(dataset_sorted);
    delete_count = 0;

    for i = 1 : patterns_count
        if (mod(i, 2) != 0)
        	dataset_sorted(i - delete_count, :) = [];
        	delete_count++;
        endif
    end

	ret = dataset_sorted;
end
