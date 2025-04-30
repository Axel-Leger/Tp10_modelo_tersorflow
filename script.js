let modelo;

async function entrenarModelo() {
  // 1. Crear modelo secuencial
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  modelo.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  // 2. Crear los datos de entrenamiento
  const x = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const y = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10.3], [9, 1]); // y = 2x + 6

  // 3. Entrenar el modelo
  await modelo.fit(x, y, {
    epochs: 350,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Época ${epoch + 1}: pérdida = ${logs.loss.toFixed(4)}`);
      },
      onTrainEnd: () => {
        document.getElementById("estado").innerText = "✅ Modelo entrenado y listo para usar";
        document.getElementById("prediccion").style.display = "block";
      }
    }
  });
}

function predecir() {
  const input = parseFloat(document.getElementById("inputX").value);
  const tensor = tf.tensor2d([input], [1, 1]);

  const resultado = modelo.predict(tensor);
  resultado.array().then(array => {
    const y = array[0][0].toFixed(2);
    document.getElementById("resultado").innerText = `Resultado: Y = ${y}`;
  });
}

// Iniciar el entrenamiento al cargar
entrenarModelo();
