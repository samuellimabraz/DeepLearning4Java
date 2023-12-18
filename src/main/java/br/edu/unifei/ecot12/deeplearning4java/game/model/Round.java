package br.edu.unifei.ecot12.deeplearning4java.game.model;


import java.util.Timer;
import java.util.TimerTask;

public class Round {
    private final String category;
    private int time;
    private Timer timer;

    private RoundListener listener;

    public Round(String category, int time) {
        this.category = category;
        this.time = time;
        this.timer = new Timer();
    }

    public void setListener(RoundListener listener) {
        this.listener = listener;
    }

    public String getCategory() {
        return category;
    }

    public int getTime() {
        return time;
    }

    public Timer getTimer() {
        return timer;
    }

    public void setTime(int time) {
        this.time = time;
    }

    public void start() {
        if (timer != null) {
            timer.cancel();
        }
        timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                time--;
                if (listener != null) {
                    listener.onTimeUpdated(time);
                }
                if (time == 0) {
                    timer.cancel();
                }
            }
        }, 0, 1000);
    }
}