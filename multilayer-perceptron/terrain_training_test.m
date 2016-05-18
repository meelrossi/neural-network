
% example of use: terrain_test(@tanh_ft, @tanh_ft_der, 0.5, 1, true)
% example of use: terrain_test(@tanh_ft, @tanh_ft_der, 0.2, 0.5, false)
function ret = terrain_test(g, g_der, n, b, incremental)
    data_filename = 'terrain8modif.txt';

    training_set = get_training_set(data_filename);
    maximum = max(max(training_set));

    inputs = [training_set(:, 1) training_set(:, 2)];
    s = [training_set(:, 3)];

    t{1} = inputs./maximum;
    t{2} = s./maximum;

    err = 0.1;

    nets = generate_nets([2 5 2 1]);

    % training net with selected training_set
    if (incremental)
        resolved_nets = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, b);
    else
        resolved_nets = multilayer_perceptron_batch(nets, t, err, g, g_der, n, b);
    endif

    % now that the net is trained with the training_set lets
    % see which output generates for the complete data set.
    complete_data_set = load('-ascii', data_filename);
    complete_data_set_inputs = [complete_data_set(:, 1) complete_data_set(:, 2)];
    complete_data_set_maximum = max(max(complete_data_set));

    layer_outputs = forward_step(complete_data_set_inputs./complete_data_set_maximum, resolved_nets, g, b);

    ret = (layer_outputs(size(resolved_nets)(2)){1}).*complete_data_set_maximum;

    saving = [complete_data_set_inputs ret];
    writeToFile(saving);
end

function ret = writeToFile(matrix)
    fid = fopen('output.txt', 'w+');
    fprintf(fid, '%f ', rows(matrix));
    fprintf(fid, '\n');
    fprintf(fid, '%f ', 0);
    fprintf(fid, '\n');
    for i = 1 : rows(matrix)
        fprintf(fid, '%f ', matrix(i,:));
        fprintf(fid, '\n');
    end
    fclose(fid);
    % csvwrite('output.txt', matrix);
end
