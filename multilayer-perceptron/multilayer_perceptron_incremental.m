
function ret = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, b)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);
    steps = 0;

    nets_count = size(nets)(2);

    % forward step
    V = forward_step(inputs, nets, g, b);

    c_error = get_error(nets_count, s, V);

    while (c_error > err)

        inputs_order = randperm(inputs_count);
        for p = 1 : inputs_count

            next_pattern = inputs_order(p);
            V = forward_step(inputs(next_pattern, :), nets, g, b);

            % back propagation
            delta{nets_count} = g_der(V{nets_count}, b).*(s(next_pattern, :) - V{nets_count});
            for i = nets_count : (-1) : 2
                % removing umbral values
                aux = nets{i}(2 : end, :);
                delta{i - 1} = g_der(V{i - 1}, b).*(delta{i} * aux');
                nets{i} = nets{i} + n * [-1 V{i - 1}]' * delta{i};
            end
            nets{1} = nets{1} + n * [-1 t{1}(next_pattern, :)]' * delta{1};

        end

        % forward step
        V = forward_step(inputs, nets, g, b);

        c_error = get_error(nets_count, s, V);
        %fflush(1);

        steps++;
    end

    steps
    ret = nets;
end

function err = get_error (nets_count, s, V)
    outputs_diff = s - V{nets_count};

    err = sum(outputs_diff.^2) / (2 * nets_count);
end
