package fr.duchess.service;

import com.datastax.spark.connector.japi.CassandraRow;
import fr.duchess.model.ActivityType;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;


public class FeatureService {

    public static Vector computeFeatures(JavaRDD<CassandraRow> data) {


        double[] features = new double[15];

        if (data.count() > 0) {

            JavaRDD<double[]> accelerationData = data.map(CassandraRow::toMap).map(row -> new double[]{(double) row.get("x"), (double) row.get("y"), (double) row.get("z")});
            JavaRDD<Vector> vectorsXYZ = accelerationData.map(Vectors::dense);
            JavaRDD<long[]> timestampAndY = data.map(CassandraRow::toMap).map(entry -> new long[]{(long) entry.get("timestamp"), ((Double) entry.get("y")).longValue()});

            //Extract features
            FeatureUtils feature = new FeatureUtils(vectorsXYZ);

            // the average acceleration
            double[] mean = feature.computeMean();

            // the variance (between sitting and standing)
            double[] variance = feature.computeVariance();

            // the standard deviation
            double[] standardDeviation = FeatureUtils.computeStandardDeviation(accelerationData, mean);

            // the average absolute difference
            double[] avgAbsDiff = feature.computeAvgAbsDifference(accelerationData, mean);

            // the average resultant acceleration
            double resultant = feature.computeResultantAcc(accelerationData);

            // the average time between peaks (walking and jogging)
            double avgTimePeak = feature.computeAvgTimeBetweenPeak(timestampAndY);

            // the average difference between X and Y
            double difference = feature.computeDifferenceBetweenAxes(mean);

            //Create feature
            features = new double[]{
                    mean[0],
                    mean[1],
                    mean[2],
                    variance[0],
                    variance[1],
                    variance[2],
                    standardDeviation[0],
                    standardDeviation[1],
                    standardDeviation[2],
                    avgAbsDiff[0],
                    avgAbsDiff[1],
                    avgAbsDiff[2],
                    resultant,
                    avgTimePeak,
                    difference
            };


        }

        return Vectors.dense(features);
    }

    public static String predict(DecisionTreeModel model, Vector feature) {

        double prediction = model.predict(feature);

        return ActivityType.fromPrediction((int) prediction);
    }
}
