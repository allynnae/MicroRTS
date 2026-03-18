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

public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;
    private final StrategyController controller = new StrategyController();
    private int lastUpdateFrame = -1;

    public NickMCTS(UnitTypeTable utt) {
        // Deep search (50) and low exploration (0.02) to ensure it commits to an attack path
        super(160, -1, 100, 50, 0.02f, 0.0f, 0.4f,
              new WorkerRush(utt), 
              new MyEvaluation(utt, null), 
              true);
        this.utt = utt;
        ((MyEvaluation)this.ef).setController(this.controller);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        // Frequency increased to every 200 frames for more responsive tactical shifts
        if (gs.getTime() % 200 == 0 && gs.getTime() != lastUpdateFrame) {
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
    private static final String OLLAMA_HOST = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    
    public volatile float aggression = 1.2f; // Default slightly aggressive
    public volatile float threatWeight = 0.8f; 
    public volatile float resourceWeight = 0.15f;
    public volatile float offensiveWeight = 0.8f; 

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(800))
            .build();
    private final Gson gson = new Gson();

    public void updateStrategy(GameState gs, int player) {
        String stateSummary = summarizeState(gs, player);
        
        // REFINED PROMPT: Encourages breaking stalemates by identifying when we have the advantage
        String prompt = "MicroRTS Battle Context: " + stateSummary + 
                        ". Task: Break the draw. If Me > En, set agg > 2.0 and off > 1.5. " +
                        "If En > Me, set thr > 3.0. Respond ONLY JSON: " +
                        "{\"agg\":float(0.5-3), \"thr\":float(0-5), \"res\":float(0-1), \"off\":float(0-2)}";

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
        return String.format("Me[Units:%d,Base:%d,Barracks:%d,Gold:%d] En[Units:%d,Base:%d,Barracks:%d,Gold:%d]",
            (my[0]+my[1]), my[2], my[3], gs.getPlayer(player).getResources(),
            (en[0]+en[1]), en[2], en[3], gs.getPlayer(1-player).getResources());
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
            JsonObject topObj = JsonParser.parseString(responseBody).getAsJsonObject();
            String modelOutput = topObj.get("response").getAsString();
            JsonObject strategy = JsonParser.parseString(modelOutput).getAsJsonObject();

            if (strategy.has("agg")) this.aggression = strategy.get("agg").getAsFloat();
            if (strategy.has("thr")) this.threatWeight = strategy.get("thr").getAsFloat();
            if (strategy.has("res")) this.resourceWeight = strategy.get("res").getAsFloat();
            if (strategy.has("off")) this.offensiveWeight = strategy.get("off").getAsFloat();
        } catch (Exception e) {}
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
        float agg = (sc != null) ? sc.aggression : 1.2f;
        float thr = (sc != null) ? sc.threatWeight : 0.8f;
        float res = (sc != null) ? sc.resourceWeight : 0.15f;
        float off = (sc != null) ? sc.offensiveWeight : 0.8f;

        // 1. Lanchester with Aggression Bias
        // We multiply the score significantly if we are in an 'aggressive' state
        float baseScore = super.evaluate(maxplayer, minplayer, gs);
        if (baseScore > 0) baseScore *= agg; 

        // 2. Global Offensive Pressure
        // Constant pull toward the enemy to prevent "dancing" in place
        float offensiveBonus = calculateGlobalOffensiveBonus(maxplayer, gs) * off;
        
        // 3. Selective Threat Penalty
        // Only care about threats if they are EXTREMELY close to home
        float threatPenalty = calculateThreat(maxplayer, gs, 0.02f) * thr;
        
        // 4. Resource carrying
        float carryingBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == maxplayer && u.getResources() > 0) {
                carryingBonus += res;
            }
        }

        return baseScore - threatPenalty + carryingBonus + offensiveBonus;
    }

    private float calculateThreat(int player, GameState gs, float weight) {
        float threatPenalty = 0.0f;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().name.equals("Base")) {
                for (Unit e : pgs.getUnits()) {
                    if (e.getPlayer() == 1 - player) {
                        int dist = Math.abs(u.getX() - e.getX()) + Math.abs(u.getY() - e.getY());
                        // Reduced threat range (7) so it doesn't get scared too easily
                        if (dist < 7) threatPenalty += (7 - dist) * weight;
                    }
                }
            }
        }
        return threatPenalty;
    }

    private float calculateGlobalOffensiveBonus(int player, GameState gs) {
        float bonus = 0;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int mapDim = pgs.getWidth() + pgs.getHeight();
        
        Unit enemyTarget = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == 1 - player) {
                // Aim for the base first to force the win
                if (enemyTarget == null || u.getType().name.equals("Base")) {
                    enemyTarget = u;
                }
            }
        }

        if (enemyTarget != null) {
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() == player && u.getType().canAttack) {
                    int dist = Math.abs(u.getX() - enemyTarget.getX()) + Math.abs(u.getY() - enemyTarget.getY());
                    // Higher reward for being close to the enemy base/units
                    bonus += (mapDim - dist) * 0.15f; 
                }
            }
        }
        return bonus;
    }
}
