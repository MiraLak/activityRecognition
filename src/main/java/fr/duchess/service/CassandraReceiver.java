package fr.duchess.service;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import fr.duchess.model.PredictionResult;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.streaming.receiver.Receiver;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;
import static com.datastax.spark.connector.japi.CassandraJavaUtil.mapToRow;
import static fr.duchess.service.PredictionService.ACCELERATION_TOTAL;
import static fr.duchess.service.PredictionService.RANDOM_FOREST_PREDICTION_MODEL;


public class CassandraReceiver extends Receiver<String>{

    private static JavaSparkContext sc;


    public CassandraReceiver(StorageLevel storageLevel,JavaSparkContext sc) {
        super(storageLevel);
        this.sc = sc;
    }

    @Override
    public StorageLevel storageLevel() {
        return StorageLevel.MEMORY_ONLY();
    }

    @Override
    public void onStart() {
        // Start the thread that receives data over a connection
        new Thread()  {
            @Override public void run() {
                receive();
            }
        }.start();
    }

    @Override
    public void onStop() {

    }

    /**
     * Compute features on latest saved accelerations, predict activity and store result.
     */
    private void receive() {
        try {

            CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable("activityrecognition", "acceleration");

            //DecisionTreeModel model = DecisionTreeModel.load(sc.sc(), DECISION_TREE_PREDICTION_MODEL);
            RandomForestModel model = RandomForestModel.load(sc.sc(), RANDOM_FOREST_PREDICTION_MODEL);

            // Until stopped or connection broken continue reading
            while (!isStopped() && !cassandraRowsRDD.rdd().isEmpty()){

                JavaRDD<CassandraRow> data = CassandraQueriesUtils.getLatestAccelerations(cassandraRowsRDD, "TEST_USER", ACCELERATION_TOTAL);
                Vector feature = FeatureService.computeFeatures(data);

                //Get prediction result
                final String predict = FeatureService.predict(model, feature);

                List<PredictionResult> predictions = new ArrayList();
                predictions.add(new PredictionResult("TEST_USER",  new Date().getTime(), predict));
                 JavaRDD<PredictionResult> rdd = sc.parallelize(predictions);

                //Save prediction result
                javaFunctions(rdd).writerBuilder("activityrecognition", "result", mapToRow(PredictionResult.class)).saveToCassandra();
                store(predict);
            }

            // Restart in an attempt to connect again when server is active again
            restart("Trying to connect again");

        } catch(Throwable t) {
            // restart if there is any other error
            restart("Error receiving data", t);
        }
    }
}
