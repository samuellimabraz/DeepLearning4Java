package br.edu.unifei.ecot12.deeplearning4java.game.model.database;

import java.util.HashMap;
import java.util.Map;

public class InputDataDaoProxy implements IInputDataDao {
    private IInputDataDao dao;
    private Map<Integer, InputData> cache;

    private static int idCounter = 0; // Contador estático para gerar IDs únicos

    public InputDataDaoProxy(IInputDataDao dao) {
        this.dao = dao;
        this.cache = new HashMap<>();
    }

    @Override
    public void save(InputData data) {
        data.setId(idCounter++);
        dao.save(data);
        cache.put(data.getId(), data);
    }

    @Override
    public InputData load(int id) {
        if (cache.containsKey(id)) {
            return cache.get(id);
        } else {
            InputData data = dao.load(id);
            cache.put(id, data);
            return data;
        }
    }
}
