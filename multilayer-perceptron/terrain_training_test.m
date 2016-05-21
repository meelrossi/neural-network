
% rand("seed", 1) set good seed

% nets = generate_nets([2 15 1]);
% example of use: terrain_training_test([2 20 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 1, 1, false)
% example of use: terrain_training_test([2 4 1], 0.001, @tanh_ft, @tanh_ft_der, 0.3, 0.2, 1, 2, false, 0.9)
% example of use: terrain_training_test([2 15 1], 0.001, @tanh_ft, @tanh_ft_der, 0.5, 0.4, 1, 3, false, 0.9, 0.2, 0.05, 11)

% example of use: terrain_training_test([2 18 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.5, 2, 1, false)
% example of use: terrain_training_test([2 9 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 2, 2, false, 0.9)
% example of use: terrain_training_test([2 11 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 2, 3, false, 0.9, 0.2, 0.1, 5)

% learningType: 1 -> Batch, 2 -> Incremental
% algorithm: 1 -> Original, 2 -> Momentum, 3 -> Adaptative etha
% K: number of positive steps befour changing etha on adaptative etha algorithm

function ret = terrain_training_test(net_structure, err, g, g_der, n, betha, learningType, algorithm, graphics, alpha = 0, a = 0, b = 0, K = 0)
    algorithms = {
                    {
                     @multilayer_perceptron_batch,
                     @multilayer_perceptron_batch_momentum,
                     @multilayer_perceptron_batch_adaptative_etha
                    }
                    {
                     @multilayer_perceptron_incremental,
                     @multilayer_perceptron_incremental_momentum,
                     @multilayer_perceptron_incremental_adaptative_etha
                    }
                };

    data_filename = 'terrain8modif.txt';
    rand('seed', 1);

    training_set = get_training_set(data_filename);
    maximum = max(max(training_set));

    inputs = [training_set(:, 1) training_set(:, 2)];
    s = [training_set(:, 3)];

    t{1} = inputs./maximum;
    t{2} = s./maximum;

    nets = generate_nets(net_structure);

    % training net with selected training_set
    fun = algorithms{learningType}{algorithm};
    resolved_nets = fun(nets, t, err, g, g_der, n, betha, graphics, alpha, a, b, K);

    % now that the net is trained with the training_set lets
    % see which output generates for the complete data set.
    complete_data_set = load('-ascii', data_filename);
    complete_data_set_inputs = [complete_data_set(:, 1) complete_data_set(:, 2)];
    complete_data_set_maximum = max(max(complete_data_set));

    layer_outputs = forward_step(complete_data_set_inputs./complete_data_set_maximum, resolved_nets, g, betha);

    ret = (layer_outputs(size(resolved_nets)(2)){1}).*complete_data_set_maximum;

    saving = [complete_data_set_inputs ret];
    writeToFile(saving);

    complete_data_set_error = get_error(size(nets)(2), complete_data_set(:, 3)./complete_data_set_maximum, layer_outputs)
end

function ret = writeToFile(matrix)
    fid = fopen('output.txt', 'w+');
    fprintf(fid, '%d ', rows(matrix));
    fprintf(fid, '\n');
    fprintf(fid, '%d ', 0);
    fprintf(fid, '\n');
    for i = 1 : rows(matrix)
        fprintf(fid, '%f ', matrix(i,:));
        fprintf(fid, ' 1 0 0 \n');
    end
    fclose(fid);
    % csvwrite('output.txt', matrix);
end
