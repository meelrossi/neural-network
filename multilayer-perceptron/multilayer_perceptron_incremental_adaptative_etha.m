
function ret = multilayer_perceptron_incremental_adaptative_etha(nets, t, err, g, g_der, n, betha, graphics, alpha, a, b, K)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);
    steps = 0;

    nets_count = size(nets)(2);
    previous_nets = nets;
    positive_steps = 0;
    old_alpha = alpha;
    test_step = false;

    % forward step
    V = forward_step(inputs, nets, g, betha);

    c_error = get_error(nets_count, s, V);
    p_error = c_error;

    if (graphics)
        f1 = figure(1);

        f2 = figure(2);

        figure(f1);
        plot(0,c_error, 'color', 'r', 'markersize', 7);
        title('Error variation', 'fontsize', 20, 'fontname', 'avenir next');
        xlabel('Step', 'fontsize', 15, 'fontname', 'avenir next');
        ylabel('Error', 'fontsize', 15, 'fontname', 'avenir next');
        vh1 = get(gca,'children');
        error_x(1)= 0;
        error_y(1)= c_error;

        figure(f2);
        plot(0,n, 'color', 'g', 'markersize', 7);
        title('Etha variation', 'fontsize', 20, 'fontname', 'avenir next');
        xlabel('Step', 'fontsize', 15, 'fontname', 'avenir next');
        ylabel('Etha', 'fontsize', 15, 'fontname', 'avenir next');
        vh2 = get(gca, 'children');
        etha_x(1) = 0;
        etha_y(1) = n;
    endif

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
            deltaW = n * [-1 t{1}(next_pattern, :)]' * delta{1} + alpha * old_deltaW{1};
            nets{1} = nets{1} + deltaW;
            old_deltaW{1} = deltaW;

        end

        % forward step
        V = forward_step(inputs, nets, g, betha);

        c_error = get_error(nets_count, s, V)
        fflush(1);

        delta_error = p_error - c_error;

        if (test_step)
            alpha = old_alpha;
        endif

        if(delta_error >= 0)
            p_error = c_error;
            positive_steps++;
            previous_nets = nets;
            test_step = false;
        else
            if(test_step)
                nets = previous_nets;
                V = forward_step(inputs, nets, g, betha);
                c_error = get_error(nets_count, s, V);
                n -= b * n;
                test_step = false;
            else
                positive_steps = 0;
                alpha = 0;
                test_step = true;
            endif
        endif
        if (positive_steps >= K)
            n += a;
        endif
        steps++;

        if (graphics)
            error_x(end + 1) = steps;
            error_y(end + 1) = c_error;
            etha_x(end + 1) = steps;
            etha_y(end + 1) = n;

            figure(f1);
            set(vh1, 'xdata', error_x, 'ydata', error_y);

            figure(f2);
            set(vh2, 'xdata', etha_x, 'ydata', etha_y);
        endif
    end

    steps
    ret = nets;
end
