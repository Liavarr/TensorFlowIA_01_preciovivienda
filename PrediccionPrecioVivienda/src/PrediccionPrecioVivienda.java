import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
//import org.tensorflow.op.Operation;
import org.tensorflow.DataType;
import org.tensorflow.Operation;

public class PrediccionPrecioVivienda {

    public static Graph crearGrafo() {
        Graph grafo = new Graph();

        // Marcadores de posición para las características de entrada: tamaño y número de habitaciones
        Operation tamano = grafo.opBuilder("Placeholder", "tamano")
                .setAttr("dtype", DataType.fromClass(Double.class))
                .build();

        Operation habitaciones = grafo.opBuilder("Placeholder", "habitaciones")
                .setAttr("dtype", DataType.fromClass(Double.class))
                .build();

        // Pesos para las características
        Operation pesoTamano = grafo.opBuilder("Const", "pesoTamano")
                .setAttr("dtype", DataType.fromClass(Double.class))
                .setAttr("value", Tensor.create(200.0)) // Peso de ejemplo para tamaño
                .build();

        Operation pesoHabitaciones = grafo.opBuilder("Const", "pesoHabitaciones")
                .setAttr("dtype", DataType.fromClass(Double.class))
                .setAttr("value", Tensor.create(5000.0)) // Peso de ejemplo para habitaciones
                .build();

        // Término de sesgo
        Operation sesgo = grafo.opBuilder("Const", "sesgo")
                .setAttr("dtype", DataType.fromClass(Double.class))
                .setAttr("value", Tensor.create(10000.0)) // Sesgo de ejemplo
                .build();

        // Combinación lineal: precio = tamano * pesoTamano + habitaciones * pesoHabitaciones + sesgo
        Operation tamanoPonderado = grafo.opBuilder("Mul", "tamanoPonderado")
                .addInput(tamano.output(0))
                .addInput(pesoTamano.output(0))
                .build();

        Operation habitacionesPonderadas = grafo.opBuilder("Mul", "habitacionesPonderadas")
                .addInput(habitaciones.output(0))
                .addInput(pesoHabitaciones.output(0))
                .build();

        Operation precioAntesDeSesgo = grafo.opBuilder("Add", "precioAntesDeSesgo")
                .addInput(tamanoPonderado.output(0))
                .addInput(habitacionesPonderadas.output(0))
                .build();

        grafo.opBuilder("Add", "precio")
                .addInput(precioAntesDeSesgo.output(0))
                .addInput(sesgo.output(0))
                .build();

        return grafo;
    }

    public static Object ejecutarGrafo(Graph grafo, Double tamano, Double habitaciones) {
        Object resultado;
        try (Session sesion = new Session(grafo)) {
            // Ejecuta el grafo con los valores de entrada
            resultado = sesion.runner()
                    .fetch("precio")
                    .feed("tamano", Tensor.create(tamano))
                    .feed("habitaciones", Tensor.create(habitaciones))
                    .run()
                    .get(0)
                    .expect(Double.class)
                    .doubleValue();
        }
        return resultado;
    }

    public static void main(String[] args) {
        Graph grafo = PrediccionPrecioVivienda.crearGrafo();
        Double tamano = 120.0; // tamaño de ejemplo en metros cuadrados
        Double habitaciones = 3.0; // número de habitaciones de ejemplo

        Object resultado = PrediccionPrecioVivienda.ejecutarGrafo(grafo, tamano, habitaciones);
        System.out.println("Precio de vivienda predicho: " + resultado);

        grafo.close();
    }
}
