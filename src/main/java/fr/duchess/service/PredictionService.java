package fr.duchess.service;

import com.datastax.spark.connector.japi.CassandraJavaUtil;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import fr.duchess.model.PredictionResult;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.*;

import java.util.*;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;
import static com.datastax.spark.connector.japi.CassandraJavaUtil.mapToRow;

public class PredictionService {

    public static final String DECISION_TREE_PREDICTION_MODEL = "predictionModel/trainingOne";
    public static final String RANDOM_FOREST_PREDICTION_MODEL = "predictionModel/RandomForest/trainingOne";
    public static final long ROWS_NUMBER = 400l; //need 100 for static activity and 400 for moving
    public static final String KEYSPACE = "activityrecognition";
    public static final String RESULT_TABLE = "result";
    public static final String TEST_USER = "TEST_USER";
    public static final String ACCELERATION_TABLE = "acceleration";


    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf()
                .setAppName("User's physical activity recognition")
                .set("spark.cassandra.connection.host", "127.0.0.1")
                .setMaster("local[*]");

        //Choose one of the prediction method below to launch
        //===================================================

        predictWithRealTimeStreaming(sparkConf);
        predictEachFiveSeconds(sparkConf, DECISION_TREE_PREDICTION_MODEL);
        predictOnce(sparkConf, DECISION_TREE_PREDICTION_MODEL);

    }

    /**
     * Predict activity and store result on Cassandra
     * @param javaSparkContext
     * @param modelName
     */
    private static void predictActivity(JavaSparkContext javaSparkContext, String modelName) {
        String predictionResult = predict(javaSparkContext, modelName);

        List<PredictionResult> predictions = new ArrayList();
        predictions.add(new PredictionResult(TEST_USER,  new Date().getTime(), predictionResult));
        JavaRDD<PredictionResult> rdd = javaSparkContext.parallelize(predictions);

        //Write result into Cassandra
        javaFunctions(rdd).writerBuilder(KEYSPACE, RESULT_TABLE, mapToRow(PredictionResult.class)).saveToCassandra();
    }

    /**
     * Predict activity using realtime streaming
     * @param sparkConf
     */
    private static void predictWithRealTimeStreaming(SparkConf sparkConf) {
        JavaStreamingContext ssc = new JavaStreamingContext(sparkConf, Durations.seconds(5));

        JavaReceiverInputDStream<String> cassandraReceiver = ssc.receiverStream(new CassandraReceiver(StorageLevel.MEMORY_ONLY(), ssc.sparkContext()));
        cassandraReceiver.print();
        System.out.println("Predicted activity = " + cassandraReceiver.toString());

        ssc.start();
        ssc.awaitTermination(); // Wait for the computation to terminate
    }

    /**
     * Predict activity
     * @param sc
     * @param modelName
     * @return prediction result
     */
    public static String predict(JavaSparkContext sc, String modelName) {

        DecisionTreeModel model = DecisionTreeModel.load(sc.sc(), modelName);

        // retrieve latestAccelerations from Cassandra and create an CassandraRDD
        CassandraJavaRDD<CassandraRow> cassandraRowsRDD = CassandraJavaUtil.javaFunctions(sc).cassandraTable(KEYSPACE, ACCELERATION_TABLE);
        JavaRDD<CassandraRow> latestAccelerations = CassandraQueriesUtils.getLatestAccelerations(cassandraRowsRDD,TEST_USER,ROWS_NUMBER);

        //Compute features and predict activity
        Vector features = FeatureService.computeFeatures(latestAccelerations);
        return FeatureService.predict(model,features);
    }

    /**
     * Predict activity each 5 seconds.
     * This is used to compare with SparkStreaming
     * @param sparkConf
     */
    private static void predictEachFiveSeconds(SparkConf sparkConf, String modelName) {
        final JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        //Never stop until we kill the service
        while(true){
            try {
                predictActivity(javaSparkContext, modelName);
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                System.out.println("Prediction failed "+e.getMessage());
            }
        }
    }

    /**
     * Predict activity without realtime streaming.
     * @param sparkConf
     */
    private static void predictOnce(SparkConf sparkConf, String modelName) {
        final JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        predictActivity(javaSparkContext, modelName);
    }

}