let modelo;

async function entrenarModelo() {
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  modelo.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  const x = tf.tensor2d([-9 ,-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [21, 1]);
  const y = tf.tensor2d([-12 ,-10, -8 ,-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], [21, 1]);


  await modelo.fit(x, y, {
    epochs: 400,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Época ${epoch +1}: pérdida = ${logs.loss.toFixed(4)}`);
      },
      onTrainEnd: () => {
        document.getElementById("prediccion").style.display = "block";
      }
    }
  });
}

entrenarModelo();

function predecir() {
  const input = parseFloat(document.getElementById("inputX").value);
  const tensor = tf.tensor2d([input], [1, 1]);

  const resultado = modelo.predict(tensor);
  resultado.array().then(array => {
    const y = array[0][0].toFixed(2);
    document.getElementById("resultado").innerText = `Resultado: Y = ${y}`;
  });
}


