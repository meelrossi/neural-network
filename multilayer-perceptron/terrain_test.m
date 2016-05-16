
% example of use: terrain_test(@tanh_ft, @tanh_ft_der, 0.5, 1, true)
% example of use: terrain_test(@tanh_ft, @tanh_ft_der, 0.2, 0.5, false)
function ret = terrain_test(g, g_der, n, b, incremental)
    data_set = load('-ascii', 'terrain8modif.txt');
    maximum = max(max(data_set));

    inputs = [data_set(:, 1) data_set(:, 2)];
    s = [data_set(:, 3)];

    t{1} = inputs./maximum;
    t{2} = s./maximum;

    err = 0.1;

    nets = generate_nets([2 5 2 1]);

    if (incremental)
        resolved_nets = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, b);
    else
        resolved_nets = multilayer_perceptron_batch(nets, t, err, g, g_der, n, b);
    endif

    layer_outputs = forward_step(inputs./maximum, resolved_nets, g, b);

    ret = (layer_outputs(size(resolved_nets)(2)){1}).*maximum;

    saving = [inputs ret];
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
