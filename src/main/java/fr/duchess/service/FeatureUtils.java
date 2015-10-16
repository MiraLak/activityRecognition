package fr.duchess.service;


import com.datastax.spark.connector.japi.CassandraRow;
import fr.duchess.model.Acceleration;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

public class FeatureUtils {

    private MultivariateStatisticalSummary summary;

    public FeatureUtils(JavaRDD<Vector> data) {
        this.summary = Statistics.colStats(data.rdd());
    }

    /**
     * @return array [ (1 / n ) * ∑ |b - mean_b|, for b in {x,y,z} ]
     */
    public static double[] computeAvgAbsDifference(JavaRDD<double[]> data, double[] mean) {

        // for each point x compute x - mean
        // then apply an absolute value: |x - mean|
        JavaRDD<Vector> abs = data.map(acceleration -> new double[]{Math.abs(acceleration[0] - mean[0]),
                Math.abs(acceleration[1] - mean[1]),
                Math.abs(acceleration[2] - mean[2])})
                .map(Vectors::dense);

        // And to finish apply the mean: for each axis (1 / n ) * ∑ |b - mean|
        return Statistics.colStats(abs.rdd()).mean().toArray();

    }

    /**
     * @return Double resultant = 1/n * ∑ √(x² + y² + z²)
     */
    public static double computeResultantAcc(JavaRDD<double[]> data) {
        // first let's compute the square of each value and the sum
        // compute then the root square: √(x² + y² + z²)
        // to finish apply a mean function: 1/n * sum [√(x² + y² + z²)]
        JavaRDD<Vector> squared = data.map(acceleration -> Math.pow(acceleration[0], 2)
                + Math.pow(acceleration[1], 2)
                + Math.pow(acceleration[2], 2))
                .map(Math::sqrt)
                .map(sum -> Vectors.dense(new double[]{sum}));

        return Statistics.colStats(squared.rdd()).mean().toArray()[0];

    }

    /**
     * @return Double[] standard deviation  = √ 1/n * ∑ (x - u)² with u = mean x
     */
    public static double[] computeStandardDeviation(JavaRDD<double[]> data, double[] mean){
        // first let's compute the difference between each value and the mean
        // apply a mean function: 1/n * [(x- mean_x)]
        // compute then the root square: √( 1/n * [(x- mean_x)])
        JavaRDD<Vector> squared = data.map(acceleration -> new double[]{
                Math.pow(acceleration[0] - mean[0],2),
                Math.pow(acceleration[1] - mean[1],2),
                Math.pow(acceleration[2] - mean[2],2)})
                .map(sum -> Vectors.dense(sum));

        double[] meanDiff = Statistics.colStats(squared.rdd()).mean().toArray();

        if(meanDiff.length>0){

            return new double[]{Math.sqrt(meanDiff[0]), Math.sqrt(meanDiff[1]), Math.sqrt(meanDiff[2])};
        }
        return new double[]{0.0,0.0,0.0};
    }

    /**
     * Compute difference between mean_x and mean_y
     * @param mean
     * @return
     */
    public double computeDifferenceBetweenAxes(double[] mean){
        return mean[0] - mean[1];
    }

    /**
     * @return array (mean_x, mean_y, mean_z)
     */
    public double[] computeMean() {
        return this.summary.mean().toArray();
    }

    /**
     * @return array (var_x, var_y, var_z)
     */
    public double[] computeVariance() {
        return this.summary.variance().toArray();
    }

    /**
     * compute average time between peaks.
     */
    public Double computeAvgTimeBetweenPeak(JavaRDD<long[]> data) {
        // define the maximum
        double[] max = this.summary.max().toArray();

        // keep the timestamp of data point for which the value is greater than 90% of max
        // and sort it !
        JavaRDD<Long> filtered_y = data.filter(record -> record[1] > 0.9 * max[1])
                .map(record -> record[0])
                .sortBy(time -> time, true, 1);

        if (filtered_y.count() > 1) {
            Long firstElement = filtered_y.first();
            Long lastElement = filtered_y.sortBy(time -> time, false, 1).first();

            // compute the delta between each tick
            JavaRDD<Long> firstRDD = filtered_y.filter(record -> record > firstElement);
            JavaRDD<Long> secondRDD = filtered_y.filter(record -> record < lastElement);

            JavaRDD<Vector> product = firstRDD.zip(secondRDD)
                    .map(pair -> pair._1() - pair._2())
                            // and keep it if the delta is != 0
                    .filter(value -> value > 0)
                    .map(line -> Vectors.dense(line));

            // compute the mean of the delta
            return Statistics.colStats(product.rdd()).mean().toArray()[0];
        }

        return 0.0;
    }
}
