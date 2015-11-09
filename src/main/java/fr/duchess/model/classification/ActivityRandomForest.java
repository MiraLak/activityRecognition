package fr.duchess.model.classification;


import fr.duchess.model.ActivityType;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class ActivityRandomForest {

    JavaRDD<LabeledPoint> trainingData;
    JavaRDD<LabeledPoint> testData;

    public ActivityRandomForest(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public Double createModel(JavaSparkContext sc) {

        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        int numTrees = 10;
        int numClasses = ActivityType.values().length;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        int maxDepth = 20;
        int maxBins = 32;

        // create model
        RandomForestModel model = org.apache.spark.mllib.tree.RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 12345);
        model.save(sc.sc(), "predictionModel/RandomForest/training_acceleration_3");

        // Compute classification accuracy on test data
        final long correctPredictionCount = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()))
                .filter(pl -> pl._1().equals(pl._2()))
                .count();
        Double classificationAccuracy = 1.0 * correctPredictionCount / testData.count();

        return classificationAccuracy;
    }
}
