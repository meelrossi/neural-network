
function ret = multilayer_perceptron_incremental_momentum(nets, t, err, g, g_der, n, betha, alpha)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);
    steps = 0;

    nets_count = size(nets)(2);

    % forward step
    V = forward_step(inputs, nets, g, betha);

    c_error = get_error(nets_count, s, V);

    for i = 1: nets_count
        n_size = size(nets{i});
        old_deltaW{i} = zeros(n_size(1), n_size(2));
    end

    while (c_error > err)

        inputs_order = randperm(inputs_count);
        for p = 1 : inputs_count

            next_pattern = inputs_order(p);
            V = forward_step(inputs(next_pattern, :), nets, g, betha);

            % back propagation
            delta{nets_count} = g_der(V{nets_count}, betha).*(s(next_pattern, :) - V{nets_count});
            for i = nets_count : (-1) : 2
                % removing umbral values
                aux = nets{i}(2 : end, :);
                delta{i - 1} = g_der(V{i - 1}, betha).*(delta{i} * aux');
                deltaW = n * [-1 V{i - 1}]' * delta{i} + alpha * old_deltaW{i};
                nets{i} = nets{i} + deltaW;
                old_deltaW{i} = deltaW;
            end
            deltaW = n * [-1 t{1}(next_pattern, :)]' * delta{1} + old_deltaW{1};
            nets{1} = nets{1} + deltaW;
            old_deltaW{1} = deltaW;

        end

        % forward step
        V = forward_step(inputs, nets, g, betha);

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
