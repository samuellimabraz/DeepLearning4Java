package br.edu.unifei.ecot12.deeplearning4java.game.model.database;

import javafx.scene.image.WritableImage;
import javafx.geometry.Point2D;
import java.util.List;

public class InputData {
    private final WritableImage image;
    private final List<Point2D> points;

    private String category;

    private int id;

    public InputData(WritableImage image, String category) {
        this.image = image;
        this.points = null;
        this.category = category;
    }

    public String getCategory() {
        return category;
    }

    public int getId() {
        return id;
    }

    public WritableImage getImage() {
        return image;
    }

    public List<Point2D> getPoints() {
        return points;
    }

    public void setId(int id) {
        this.id = id;
    }
}
