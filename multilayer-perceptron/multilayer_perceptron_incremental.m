
function ret = multilayer_perceptron_incremental(nets, t, err, g, g_der, n, betha, graphics)
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

    if (graphics)
        figure(1);
        plot(0,c_error);
        vh = get(gca,'children');
        y(1)= c_error;
        x(1)= 0;
    endif

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
                nets{i} = nets{i} + n * [-1 V{i - 1}]' * delta{i};
            end
            nets{1} = nets{1} + n * [-1 t{1}(next_pattern, :)]' * delta{1};

        end

        % forward step
        V = forward_step(inputs, nets, g, betha);

        c_error = get_error(nets_count, s, V)
        fflush(1);

        steps++;

        if (graphics)
            y(end + 1) = c_error;
            x(end + 1) = steps;
            fflush(1);
            set(vh, 'xdata', x, 'ydata', y); 
            pause(0.1);
        endif
    end

    steps
    ret = nets;
end
