package fr.duchess.model;

import java.io.Serializable;
import java.util.Date;
import java.util.UUID;

/**
 * Created by lakhal on 20/10/15.
 */
public class PredictionResult implements Serializable{

    private String user_id;
    private long timestamp;
    private String prediction;

    public PredictionResult(String user_id, long timestamp, String prediction) {
        this.user_id = user_id;
        this.timestamp = timestamp;
        this.prediction = prediction;
    }

    public String getUser_id() {
        return user_id;
    }

    public void setUser_id(String user_id) {
        this.user_id = user_id;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }
}
