package fr.duchess.model;


public class TimeWindow {

    private Long start;
    private Long stop;
    private long intervals;

    public TimeWindow(Long start, Long stop, long intervals) {
        this.start = start;
        this.stop = stop;
        this.intervals = intervals;
    }

    public Long getStart() {
        return start;
    }

    public void setStart(Long start) {
        this.start = start;
    }

    public Long getStop() {
        return stop;
    }

    public void setStop(Long stop) {
        this.stop = stop;
    }

    public long getIntervals() {
        return intervals;
    }

    public void setIntervals(long intervals) {
        this.intervals = intervals;
    }
}
