
function ret = get_training_set(data_filename, x_strips)

    complete_dataset = load('-ascii', data_filename);
    dataset_sorted = sortrows(complete_dataset);

    patterns_count = rows(dataset_sorted);
    delete_count = 0;

    min_x = min(dataset_sorted(:, 1));
    max_x = max(dataset_sorted(:, 1));
    min_y = min(dataset_sorted(:, 2));
    max_y = max(dataset_sorted(:, 2));

    last_x = dataset_sorted(1, 1);
    different_x = 1;

    for i = 1 : patterns_count

        x = dataset_sorted(i - delete_count, 1);
        y = dataset_sorted(i - delete_count, 2);

        if (x_strips)

            if (mod(different_x + 1, 2) == 0)
                if (x != last_x)
                    different_x++;
                    dataset_sorted(i - delete_count, :) = [];
                    delete_count++;
                endif
            else
                if (x == last_x)
                    dataset_sorted(i - delete_count, :) = [];
                    delete_count++;
                else
                    different_x++;
                endif
            endif
            last_x = x;

        else

            is_outer_bool = is_outer(min_x, max_x, min_y, max_y, x, y);

            if (mod(i, 4) != 0 && !is_outer_bool)
                dataset_sorted(i - delete_count, :) = [];
                delete_count++;
            endif

        endif
    end

    ret = dataset_sorted;
end

function bool = is_outer(min_x, max_x, min_y, max_y, x, y)
    bool = (min_x == x && min_y == y) || (min_x == x && max_y == y) || (max_x == x && min_y == y) || (max_x == x && max_y == y);
end
