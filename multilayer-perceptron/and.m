
function resp = and (inputs)
    inputs_count = rows(inputs);
    for i = 1 : inputs_count
        resp(i) = (sum(inputs(i, :)) / columns(inputs)) == 1;
    end
    resp = resp';
end
