
function ret = multilayer_perceptron_batch_adaptative_etha(nets, t, err, g, g_der, n, betha, graphics, alpha, a, b, K)
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

    p_error = c_error;

    previous_nets = nets;
    positive_steps = 0;
    old_alpha = alpha;
    test_step = false;

    if(graphics)
        f1 = figure(1);
        f2 = figure(2);

        figure(f1);
        plot(0,c_error, 'color', 'r', 'markersize', 7);
        title('Error variation', 'fontsize', 20, 'fontname', 'avenir next');
        xlabel('Step', 'fontsize', 15, 'fontname', 'avenir next');
        ylabel('Error', 'fontsize', 15, 'fontname', 'avenir next');
        vh1 = get(gca,'children');
        x(1)= 0;
        error_y(1)= c_error;

        figure(f2);
        plot(0,n, 'color', 'g', 'markersize', 7);
        title('Etha variation', 'fontsize', 20, 'fontname', 'avenir next');
        xlabel('Step', 'fontsize', 15, 'fontname', 'avenir next');
        ylabel('Etha', 'fontsize', 15, 'fontname', 'avenir next');
        vh2 = get(gca, 'children');
        etha_y(1) = n;
    endif

    for i = 1: nets_count
        n_size = size(nets{i});
        old_deltaW{i} = zeros(n_size(1), n_size(2));
    end

    while (c_error > err)
        % back propagation
        delta{nets_count} = g_der(V{nets_count}, betha).*(s - V{nets_count});
        for i = nets_count : (-1) : 2
            % removing umbral values
            aux = nets{i}(2 : end, :);
            delta{i - 1} = g_der(V{i - 1}, betha).*(delta{i} * aux');
            deltaW = n * [ones(inputs_count,1).*(-1) V{i - 1}]' * delta{i} + alpha * old_deltaW{i};
            nets{i} = nets{i} + deltaW;
            old_deltaW{i} = deltaW;
        end
        deltaW = n * [ones(inputs_count,1).*(-1) t{1}]' * delta{1}  + alpha * old_deltaW{1};
        nets{1} = nets{1} + deltaW;
        old_deltaW{1} = deltaW;

        % forward step
        V = forward_step(inputs, nets, g, betha);

        c_error = get_error(nets_count, s, V);
        %fflush(1);

        delta_error = p_error - c_error;

        if (test_step)
            alpha = old_alpha;
        endif

        if(delta_error > 0)
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
            x(end + 1) = steps;
            error_y(end + 1) = c_error;
            etha_y(end + 1) = n;

            figure(f1);
            set(vh1, 'xdata', x, 'ydata', error_y);

            figure(f2);
            set(vh2, 'xdata', x, 'ydata', etha_y);
        endif
    end

    steps
    ret = nets;
end

function err = get_error (nets_count, s, V)
    outputs_diff = s - V{nets_count};

    err = sum(outputs_diff.^2) / (2 * nets_count);
end

