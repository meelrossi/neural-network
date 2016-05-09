% i VERTICAL, j HORIZONTAL
% inputs matrix[i][j]

function net = simple_perceptron_incremental (err, t, g, n)
    inputs = t{1}; % matrix[inputs_count][input_size]
    inputs_count = rows(inputs);
    input_size = columns(inputs);
    s = t{2}; % matrix[inputs_count][s_size]
    s_size = columns(s);

    W = rand(input_size + 1, s_size).*2 .- 1; % matrix[input_size + 1][s_size]

    c_error = get_error(W, inputs, g, s);

    while (c_error > err)
        inputs_order = randperm(inputs_count);
        for i = 1 : inputs_count
            output = [-1 inputs(inputs_order(i), :)] * W;
            output_diff = s(inputs_order(i), :) - g(output);

            W = W + n * output_diff * [-1 inputs(inputs_order(i), :)]';
        end
        c_error = get_error(W, inputs, g, s);
    end

    net = W;
end

function err = get_error (W, inputs, g, s)
    inputs_count = rows(inputs);
    umbral_input = [ones(inputs_count, 1).*(-1) inputs]; % matrix[inputs_count][input_size + 1]
    outputs = umbral_input * W; % matrix[inputs_count][s_size]
    outputs_diff = s - g(outputs);

    err = sum(outputs_diff.^2) / (2 * inputs_count);
end