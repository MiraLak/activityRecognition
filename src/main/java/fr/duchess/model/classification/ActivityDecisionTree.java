package fr.duchess.model.classification;

import fr.duchess.model.ActivityType;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class ActivityDecisionTree {

    JavaRDD<LabeledPoint> trainingData;
    JavaRDD<LabeledPoint> testData;

    public ActivityDecisionTree(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
        this.trainingData = trainingData;
        this.testData = testData;
    }

    public Double createModel(JavaSparkContext sc) {

        // parameters
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numClasses = ActivityType.values().length; //num of classes = num of activity to predict
        String impurity = "gini"; //measure of the homogeneity of the labels at the node ∑Ci=1fi(1−fi)
        int maxDepth = 20;
        int maxBins = 32; //minimum value for bins

        // create model
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
        model.save(sc.sc(), "predictionModel/DecisionTree/training_acceleration_3");

        // Compute classification accuracy on test data
        final long correctPredictionCount = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()))
                                                    .filter(pl -> pl._1().equals(pl._2()))
                                                    .count();
        Double classificationAccuracy = 1.0 * correctPredictionCount / testData.count();

        return classificationAccuracy;
    }
}