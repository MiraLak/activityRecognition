package fr.duchess.service;

import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import fr.duchess.model.ActivityType;
import fr.duchess.model.classification.ActivityDecisionTree;
import fr.duchess.model.TimeWindow;
import fr.duchess.model.classification.ActivityRandomForest;
import org.apache.commons.collections.CollectionUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.*;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.javaFunctions;
import static fr.duchess.model.ActivityType.*;

public class TrainingService {

    public static final long ONE_SECOND = 100000000;
    public static final long TWO_SECONDS = 500000000l;
    private static final String KEYSPACE = "activityrecognition";
    private static final String TRAINING_TABLE = "trainingacceleration";

    private JavaSparkContext sc;

    public static void main(String[] args){

        TrainingService trainingService = new TrainingService();
        trainingService.initSparkContext();
        trainingService.createDecisionTreeWithTrainingData();

    }

    public void initSparkContext(){

        SparkConf sparkConf = new SparkConf()
                .setAppName("Activity recognition")
                .set("spark.cassandra.connection.host", "127.0.0.1")
                .setMaster("local[*]");

        sc = new JavaSparkContext(sparkConf);
    }

    public void createDecisionTreeWithTrainingData(){

        List<LabeledPoint> labeledPoints = new ArrayList<>();
        CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable(KEYSPACE, TRAINING_TABLE);

        List<String> users = CassandraQueriesUtils.getUsers(cassandraRowsRDD);

        // Create labeled points per user and activity
        for(ActivityType activity : values()) {
            users.stream().limit(10).forEach(userId -> {

                //retrieve data from DB for specific user and activity
                JavaRDD<Long> times = CassandraQueriesUtils.getTimesForUserAndActivity(cassandraRowsRDD, userId, activity);

                //create time windows with 1 second jump and 2s intervals
                if(!times.isEmpty()){
                    List<TimeWindow> timeWindows = createTimeWindows(times);
                    timeWindows.stream()
                            .forEach(timeWindow -> labeledPoints.addAll(createLabeledPointsForTimeWindow(cassandraRowsRDD, activity, userId, timeWindow, TWO_SECONDS)));
                }
            });
        }

        // create model prediction: decision tree
        if (CollectionUtils.isNotEmpty(labeledPoints)) {

            // transform into RDD
            JavaRDD<LabeledPoint> labeledPointsRdd = sc.parallelize(labeledPoints);

            // Split labeledPointsRdd into 2 sets : training (60%) and test (40%).
            JavaRDD<LabeledPoint>[] splits = labeledPointsRdd.randomSplit(new double[]{0.6, 0.4});
            JavaRDD<LabeledPoint> trainingDataSet = splits[0].cache();
            JavaRDD<LabeledPoint> testDataSet = splits[1];

            // Create DecisionTree and compute accuracy
            double decisionTreeAccuracy = new ActivityDecisionTree(trainingDataSet, testDataSet).createModel(sc);

            // Create RandomForest and compute accuracy
            double randomForestAccuracy = new ActivityRandomForest(trainingDataSet, testDataSet).createModel(sc);

            System.out.println("Labeled Points size " + labeledPointsRdd.count());
            System.out.println("Decision Tree Accuracy: " + decisionTreeAccuracy);
            System.out.println("Random Forest Accuracy: " + randomForestAccuracy);

        }
    }

    /**
     * Create LabeledPoint used in MLlib to create a predictive model
     * @param cassandraRowsRDD
     * @param activity
     * @param userId
     * @param timeWindow
     * @param frequency
     * @return labeled points for user and activityand timewindow
     */
    private List<LabeledPoint> createLabeledPointsForTimeWindow(CassandraJavaRDD<CassandraRow> cassandraRowsRDD, ActivityType activity, String userId, TimeWindow timeWindow, long frequency) {
        List<LabeledPoint> labeledPointsForTimeWindow = new ArrayList<>();

        for (int i = 0; i <= timeWindow.getIntervals(); i++) {

            JavaRDD<CassandraRow> data = CassandraQueriesUtils.getRangeDataForUserAndActivity(cassandraRowsRDD, userId, activity, timeWindow, i, frequency);

            if (data.count() > 0) {
                Vector vector = FeatureService.computeFeatures(data);
                LabeledPoint labeledPoint = createLabeledPoint(activity, vector);
                labeledPointsForTimeWindow.add(labeledPoint);
            }
        }
        return labeledPointsForTimeWindow;
    }

    /**
     * Create time windows for the timelap with a jump every second gap and splitted to intervals of 2 seconds
     * @param times
     * @return list of time window
     */
    private List<TimeWindow> createTimeWindows(JavaRDD<Long> times) {

        //compute time difference between two consecutive instant T1 and T2
        final Long startTimestamp = times.first();
        final Long endTimestamp = times.sortBy(time -> time, false, 1).first();

        JavaPairRDD<Long[], Long> timesDifferences = computeTimestampDifference(times, startTimestamp, endTimestamp);

        //define jumps intervals in time: pair of {T1,T2} where difference is over specified gap
        JavaPairRDD<Long, Long> timeJumps = timesDifferences.filter(pair -> pair._2() > ONE_SECOND)
                .mapToPair(pair -> new Tuple2<>(pair._1()[1], pair._1()[0]));

        //define windows using jumps intervals
        List<Long> timeJumpsCollection = timeJumps.flatMap(pair -> Arrays.asList(pair._1(), pair._2()))
                .sortBy(t -> t, true, 1)
                .collect();
        timeJumpsCollection.add(0, startTimestamp);
        timeJumpsCollection.add(endTimestamp);

        List<TimeWindow> windows = new ArrayList();
        Iterator<Long> jumpsIterator = timeJumpsCollection.iterator();

        while(jumpsIterator.hasNext()){
            Long startWindow = jumpsIterator.next();
            Long stopWindow = jumpsIterator.next();
            long intervals = (long) Math.round((stopWindow - startWindow) / TWO_SECONDS);

            windows.add(new TimeWindow(startWindow, stopWindow, intervals));
        }

        return windows;
    }

    /**
     * Compute difference between two successive timestamps records
     *  T2 - T1
     * @param times
     * @param startTimestamp
     * @param endTimestamp
     * @return PairRDD containing {[T2, T1], T2-T1}
     */
    private JavaPairRDD<Long[], Long> computeTimestampDifference(JavaRDD<Long> times, Long startTimestamp, Long endTimestamp) {

        JavaRDD<Long> timesWithoutStart = times.filter(time -> time > startTimestamp);
        JavaRDD<Long> timesWithoutEnd = times.filter(time -> time < endTimestamp);

        JavaPairRDD<Long[], Long> diffTimes = timesWithoutStart.zip(timesWithoutEnd)
                .mapToPair(pair -> new Tuple2<>(new Long[]{pair._1(), pair._2()}, pair._1() - pair._2()));

        return diffTimes;
    }

    /**
     * Create labeled point based on computed features vector and selected activity.
     * @param activity
     * @param vector
     * @return labeled point
     */
    private static LabeledPoint createLabeledPoint(ActivityType activity, Vector vector) {
        return new LabeledPoint((double) activity.getNumber(), vector);
    }

}