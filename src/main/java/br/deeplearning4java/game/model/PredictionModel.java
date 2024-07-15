package br.deeplearning4java.game.model;

import br.deeplearning4java.game.model.database.StringListConverter;

import javax.persistence.*;
import java.util.List;

@Entity
@Inheritance(strategy = InheritanceType.SINGLE_TABLE)
public abstract class PredictionModel {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "database_type")
    public String databaseType;
    @Column(name = "model_name")
    public String modelName;
    @Transient
    protected boolean modelLoaded = false;

    @Convert(converter = StringListConverter.class)
    protected List<String> categories;

    public abstract void loadModel();

    public abstract List<PredictionResult> predict(byte[] data);

    public List<String> getCategories() {
        return categories;
    }
}
