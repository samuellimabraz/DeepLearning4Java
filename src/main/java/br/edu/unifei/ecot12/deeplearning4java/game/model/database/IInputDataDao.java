package br.edu.unifei.ecot12.deeplearning4java.game.model.database;

public interface IInputDataDao {
    void save(InputData data);
    InputData load(int id);
}
