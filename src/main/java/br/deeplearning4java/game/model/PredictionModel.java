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

    @Convert(converter = StringListConverter.class)
    protected static List<String> categories;

    public abstract List<PredictionResult> predict(byte[] data);

    public List<String> getCategories() {
        return categories;
    }
}
