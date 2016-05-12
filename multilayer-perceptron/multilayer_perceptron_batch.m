
function ret = multilayer_perceptron_batch(nets, t, err, g, g_inv, n ,b)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);

    nets_count = size(nets)(2);

    % forward step
    in_layer = inputs;
    for i = 1 : nets_count
        layer = [ones(inputs_count, 1).*(-1) in_layer];
        V{i} = g(layer * nets{i}, b);
        in_layer = V{i};
    end

    c_error = get_error(inputs_count, s, V);

    while (c_error > err)

        % back propagation
        delta{nets_count} = g_inv(V{M}, b).*(s - V{M});
        for i = M : (-1) : 2
            % removing umbral values
            aux = nets{i}(2 : end, :);
            delta{i - 1} = g_inv(V{i - 1}, b).*(delta{i} * aux');
            nets{i} = nets{i} + n * [ones(N,1).*(-1) V{i - 1}]' * delta{i};
        end
        nets{1} = nets{1} + n * [ones(N,1).*(-1) t{1}]' * delta{1};


        % forward step
        in_layer = inputs;
        for i = 1 : nets_count
            layer = [ones(inputs_count, 1).*(-1) in_layer];
            V{i} = g(layer * nets{i}, b);
            in_layer = V{i};
        end

        c_error = get_error(inputs_count, s, V);
    end

    ret = nets;
end

function err = get_error (inputs_count, s, V)
    outputs_diff = s - V{inputs_count};

    err = sum(outputs_diff.^2) / (2 * inputs_count);
end
