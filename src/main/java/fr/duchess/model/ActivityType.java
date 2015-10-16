package fr.duchess.model;


public enum ActivityType {

    WALKING(0,"Walking"),
    JOGGING(1,"Jogging"),
    STANDING(2,"Standing"),
    SITTING(3,"Sitting");

    //Other possible activities
    //UPSTAIRS(4,"Upstairs"),
    //DOWNSTAIRS(5,"Downstairs"),
    //JUMPING(6,"Jumping"),
    //MOON_WALK(7,"Moonwalk");

    private String label;
    private int number;

    ActivityType(int number, String label) {
        this.label = label;
        this.number = number;
    }

    public int getNumber() {
        return number;
    }

    public String getLabel(){
        return label;
    }

    public static String fromPrediction(int prediction) throws IllegalArgumentException {
        try{
            return ActivityType.values()[prediction].getLabel();
        }catch( ArrayIndexOutOfBoundsException e ) {
            throw new IllegalArgumentException("No activity predicted");
        }
    }
}
