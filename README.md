# neural-network

### Clonar y ejecutar

1) Abrir una terminal.

2) Ejecutar: `git clone git@github.com:meelrossi/neural-network.git`.

3) Cambiar al directorio: `cd neural-network/multilayer-perceptron`.

4) Iniciar octave con el comando: `octave`.

5) Correr el test del terreno con el comando: `[generalization, nets] = terrain_training_test(net_structure, error, g, g_der, n, betha, learningType, algorithm, graphics, alpha, a, b, K);`, donde:
```
 - net_structure : array de enteros indicando para cada capa la cantidad de neuronas que posee.
 - error : el error máximo que debe tener el entrenamiento.
 - g : función de activación.
 - g_der : derivada de la función de activación.
 - n : valor de etha.
 - betha : parámetro utilizado en la función de activación.
 - learningType : 1 (batch) o 2 (incremental).
 - algorithm : 1 (original), 2 (momentum) o 3 (adaptative etha).
 - graphics : valor booleano indicando si graficar el error a medida que avanza elalgoritmo.
 - alpha : valor utilizado para modificación de momentum.
 - a : valor en el que se aumenta etha en modificación de adaptative etha.
 - b : porcentaje en el que se disminuye etha en modificación de adaptative etha.
 - K : cantidad de pasos positivos consecutivos antes de modificar el etha enmodificación de adaptative etha.

```
#### Ejemplo 1 - Batch original
```
[generalization, nets] = terrain_training_test([2 7 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 1, 1, false)
```

#### Ejemplo 2 - Batch momentum
```
[generalization, nets] = terrain_training_test([2 17 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 1, 2, false, 0.9)
```

#### Ejemplo 3 - Batch adaptative etha
```
[generalization, nets] = terrain_training_test([2 12 1], 0.001, @tanh_ft, @tanh_ft_der, 0.5, 0.4, 1, 3, false, 0.9, 0.2, 0.05, 11)
```

#### Ejemplo 4 - Incremental original
```
[generalization, nets] = terrain_training_test([2 11 1], 0.001, @tanh_ft, @tanh_ft_der, 0.3, 0.5, 2, 1, false)
```

#### Ejemplo 5 - Incremental momentum
```
[generalization, nets] = terrain_training_test([2 11 1], 0.001, @tanh_ft, @tanh_ft_der, 0.02, 0.5, 2, 2, false, 0.9)
```

#### Ejemplo 6 - Incremental adaptative etha
```
[generalization, nets] = terrain_training_test([2 11 1], 0.001, @tanh_ft, @tanh_ft_der, 0.02, 0.5, 2, 3, false, 0.9, 0.02, 0.1, 5)
```
