package fr.duchess.service;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import fr.duchess.model.ActivityType;
import fr.duchess.model.TimeWindow;
import org.apache.spark.api.java.JavaRDD;

import java.util.List;

public class CassandraQueriesUtils {

    public static JavaRDD<CassandraRow> getLatestAccelerations(CassandraJavaRDD<CassandraRow> cassandraRowsRDD, String user, long maxAcceleration){

        return  cassandraRowsRDD
                .select("timestamp", "x", "y", "z")
                .where("user_id=?", user)
                .withDescOrder()
                .limit(maxAcceleration)
                .repartition(1);
    }

    public static List<String> getUsers(CassandraJavaRDD<CassandraRow> cassandraRowsRDD){
        return cassandraRowsRDD
                .select("user_id").distinct()
                .map(CassandraRow::toMap)
                .map(row -> (String) row.get("user_id"))
                .collect();
    }

    public static JavaRDD<Long> getTimesForUserAndActivity(CassandraJavaRDD<CassandraRow> cassandraRowsRDD, String userId, ActivityType activity){
        return  cassandraRowsRDD
                .select("timestamp", "activity")
                .where("user_id=? AND activity=?", userId, activity.getLabel())
                .map(CassandraRow::toMap)
                .map(row -> (long) row.get("timestamp"));
    }

    public static JavaRDD<CassandraRow> getRangeDataForUserAndActivity(CassandraJavaRDD<CassandraRow> cassandraRowsRDD, String userId, ActivityType activity, TimeWindow timeWindow, int interval, long frequency){
        return cassandraRowsRDD
                .select("timestamp", "x", "y", "z")
                .where("user_id=? AND activity=? AND timestamp < ? AND timestamp > ?",
                        userId, activity.getLabel(), timeWindow.getStop() + interval * frequency, timeWindow.getStop() + (interval - 1) * frequency)
                .withAscOrder();
    }
}
