package ai.mcts.submissions.nick_mcts;

import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.LanchesterEvaluationFunction;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.ArrayList;
import java.util.List;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import rts.units.Unit;
import rts.PhysicalGameState;

/**
 * NickMCTS: An adaptive NaiveMCTS agent using environment-configurable LLM strategy.
 */
public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;
    private final StrategyController controller = new StrategyController();
    private int lastUpdateFrame = -1;

    public NickMCTS(UnitTypeTable utt) {
        super(100, -1, 100, 10, 0.3f, 0.0f, 0.4f,
              new WorkerRush(utt), 
              new MyEvaluation(utt, null), 
              true);
        this.utt = utt;
        ((MyEvaluation)this.ef).setController(this.controller);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (gs.getTime() % 400 == 0 && gs.getTime() != lastUpdateFrame) {
            lastUpdateFrame = gs.getTime();
            controller.updateStrategy(gs, player);
        }
        return super.getAction(player, gs);
    }

    @Override
    public AI clone() {
        return new NickMCTS(utt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}

class StrategyController {
    // Standard environment variables for tournament configuration
    private static final String OLLAMA_HOST = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    
    public volatile float aggression = 1.0f;
    public volatile float threatWeight = 1.0f;
    public volatile float resourceWeight = 0.2f;

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(500))
            .build();
    private final Gson gson = new Gson();

    public void updateStrategy(GameState gs, int player) {
        String stateSummary = summarizeState(gs, player);
        String prompt = "MicroRTS state: " + stateSummary + 
                        ". Respond ONLY JSON: {\"agg\":float(0.5-2), \"thr\":float(0-5), \"res\":float(0-1)}";

        // Build JSON safely using a Map and Gson to prevent injection
        Map<String, Object> payload = new HashMap<>();
        payload.put("model", OLLAMA_MODEL);
        payload.put("prompt", prompt);
        payload.put("stream", false);
        payload.put("format", "json");
        
        String jsonBody = gson.toJson(payload);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
              .thenApply(HttpResponse::body)
              .thenAccept(this::parseAndApply)
              .exceptionally(e -> null);
    }

    private String summarizeState(GameState gs, int player) {
        int[] my = countUnits(gs, player);
        int[] en = countUnits(gs, 1 - player);
        return String.format("Me[W:%d,C:%d,B:%d,Br:%d,G:%d] En[W:%d,C:%d,B:%d,Br:%d,G:%d]",
            my[0], my[1], my[2], my[3], gs.getPlayer(player).getResources(),
            en[0], en[1], en[2], en[3], gs.getPlayer(1-player).getResources());
    }

    private int[] countUnits(GameState gs, int p) {
        int w=0, c=0, b=0, br=0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == p) {
                String n = u.getType().name;
                if (n.equals("Worker")) w++;
                else if (n.equals("Base")) b++;
                else if (n.equals("Barracks")) br++;
                else c++;
            }
        }
        return new int[]{w, c, b, br};
    }

    private void parseAndApply(String responseBody) {
        try {
            // Ollama returns a wrapper JSON; the actual model output is in the "response" field
            JsonObject topObj = JsonParser.parseString(responseBody).getAsJsonObject();
            String modelOutput = topObj.get("response").getAsString();
            JsonObject strategy = JsonParser.parseString(modelOutput).getAsJsonObject();

            if (strategy.has("agg")) this.aggression = strategy.get("agg").getAsFloat();
            if (strategy.has("thr")) this.threatWeight = strategy.get("thr").getAsFloat();
            if (strategy.has("res")) this.resourceWeight = strategy.get("res").getAsFloat();
        } catch (Exception e) { /* Keep current weights */ }
    }
}

class MyEvaluation extends LanchesterEvaluationFunction {
    private StrategyController sc;

    public MyEvaluation(UnitTypeTable utt, StrategyController sc) {
        this.sc = sc;
    }

    public void setController(StrategyController sc) { this.sc = sc; }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        float agg = (sc != null) ? sc.aggression : 1.0f;
        float thr = (sc != null) ? sc.threatWeight : 1.0f;
        float res = (sc != null) ? sc.resourceWeight : 0.2f;

        float score = super.evaluate(maxplayer, minplayer, gs) * agg;
        float threatPenalty = calculateThreat(maxplayer, gs) * thr;
        
        float carryingBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == maxplayer && u.getResources() > 0) {
                carryingBonus += res;
            }
        }
        return score - threatPenalty + carryingBonus;
    }

    private float calculateThreat(int player, GameState gs) {
        float threatPenalty = 0.0f;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().name.equals("Base")) {
                for (Unit e : pgs.getUnits()) {
                    if (e.getPlayer() == 1 - player) {
                        int dist = Math.abs(u.getX() - e.getX()) + Math.abs(u.getY() - e.getY());
                        if (dist < 8) threatPenalty += (8 - dist) * 0.1f;
                    }
                }
            }
        }
        return threatPenalty;
    }
}
