/**
 * Dynamic Associative Network (DAN)
 *
 * Hierarchical temporal learning through:
 * - Pattern detection
 * - Predictive growth
 * - Dynamic pruning
 * - Astrocyte-mediated homeostatic regulation
 *
 * @author C.Rowe
 * (January 2026)
 */
package org.simbrain.custom_sims.simulations.dynamical_systems

import kotlinx.coroutines.runBlocking
import org.simbrain.custom_sims.*
import org.simbrain.network.core.Network
import org.simbrain.network.core.Neuron
import org.simbrain.network.core.Synapse
import org.simbrain.network.updaterules.LinearRule
import org.simbrain.util.place
import org.simbrain.util.point
import org.piccolo2d.nodes.PPath
import java.awt.Color


/**
 * Dynamic Associative Network (DAN)
 *  -More information: https://www.afbeavers.net/drg/about
 *
 * := Hierarchical associative learning networks that demonstrate:
 * - Temporal pattern detection across multiple layers (0-6)
 * - Unsupervised hierarchy formation
 * - Predictive accuracy-based pruning
 * - Tunable temporal credit assignment
 */
val dynamicAssociativeNetwork = newSim {
    workspace.clearWorkspace()
    val networkComponent = addNetworkComponent("Dynamic Associative Network")
    val network = networkComponent.network

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CORE PARAMETERS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    var coFireThreshold = 3       // How many times neurons must co-fire to form association
    var temporalWindow = 5        // Timesteps to look back for co-firing
    val firingThreshold = 0.5     // Activation level considered firing
    val layerSpacing = 100.0      // Visual spacing between layers

    var pruningEnabled = false    // Pruning OFF by default
    var pruningThreshold = 10     // Minimum predictions before pruning decision
    var pruningWindow = 3         // Timesteps to wait for prediction

    var timestep = 0L
    var totalNeuronsCreated = 0

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // DATA STRUCTURES
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Per-layer defaults
    val neuronsByLayer = mutableMapOf<Int, MutableList<Neuron>>()
    val decayRates = mutableMapOf(0 to 1.0, 1 to 0.9, 2 to 0.9, 3 to 0.9, 4 to 0.9, 5 to 0.9, 6 to 0.9)
    val layerPruningEnabled = mutableMapOf(1 to true, 2 to true, 3 to true, 4 to true, 5 to true, 6 to true)

    for (layer in 0..6) {
        neuronsByLayer[layer] = mutableListOf()
    }

    // Temporal tracking
    data class TimestepRecord(val timestep: Long, val firedNeuronsByLayer: Map<Int, Set<Neuron>>)
    val firingHistory = mutableListOf<TimestepRecord>()

    // Association tracking
    data class NeuronGroup(val neurons: Set<Neuron>) {
        override fun equals(other: Any?) = other is NeuronGroup && neurons == other.neurons
        override fun hashCode() = neurons.hashCode()
    }
    val alreadyCreatedAssociations = mutableSetOf<NeuronGroup>()

    // Pruning tracking
    data class PredictionRecord(
        val neuron: Neuron,
        var totalPredictions: Int = 0,
        var correctPredictions: Int = 0,
        var activePredictions: MutableList<Pair<Long, Set<Neuron>>> = mutableListOf()
    )
    val predictionRecords = mutableMapOf<Neuron, PredictionRecord>()

    // Astrocyte grid
    data class Astrocyte(
        val x: Double, val y: Double, val width: Double, val height: Double,
        var calcium: Double = 0.0,
        val decayRate: Double = 0.97
    )

    var astrocytesEnabled = true
    val astrocyteGain = 0.02
    var astrocyteCalciumDecay = 0.97
    var astrocyteSuppressionThreshold = 0.5
    var astrocyteRecoveryThreshold = 0.3
    var astrocyteWeightSuppression = 0.95
    val astrocyteWeightRecovery = 1.02
    val astrocyteTileWidth = 180.0
    val astrocyteTileHeight = 190.0
    val astrocyteTileXStarts = listOf(-300.0, -120.0, 60.0, 240.0)
    val astrocytes = mutableListOf(
        Astrocyte(x = -300.0, y = 20.0, width = 180.0, height = 190.0),
        Astrocyte(x = -120.0, y = 20.0, width = 180.0, height = 190.0),
        Astrocyte(x = 60.0,   y = 20.0, width = 180.0, height = 190.0),
        Astrocyte(x = 240.0,  y = 20.0, width = 180.0, height = 190.0)
    )
    val astrocyteNodes = mutableListOf<PPath>()
    val originalSynapseWeights = mutableMapOf<Synapse, Double>()
    var canvasLayer: org.piccolo2d.PLayer? = null

    fun addAstrocyteRow(rowY: Double) {
        astrocyteTileXStarts.forEach { tileX ->
            val astro = Astrocyte(x = tileX, y = rowY, width = astrocyteTileWidth, height = astrocyteTileHeight)
            astrocytes.add(astro)
            canvasLayer?.let { layer ->
                val rect = PPath.createRectangle(astro.x, astro.y, astro.width, astro.height)
                rect.paint = Color(255, 220, 50, 0)
                rect.strokePaint = Color(200, 180, 50, 40)
                rect.pickable = false
                layer.addChild(0, rect)
                astrocyteNodes.add(rect)
            }
        }
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HELPER FUNCTIONS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fun createNeuron(layer: Int, label: String, x: Double, y: Double): Neuron {
        return Neuron(LinearRule()).apply {
            this.label = label
            activation = 0.0
            clamped = false
            location = point(x, y)
            lowerBound = 0.0
            upperBound = 1.0
            increment = 1.0
        }
    }

    fun createAssociationNeuron(sourceNeurons: Set<Neuron>, targetLayer: Int) {
        val neurons = sourceNeurons.toList()
        val centerX = neurons.map { it.x }.average()
        val centerY = neurons.map { it.y }.average()
        val newY = centerY - layerSpacing

        val label = "L${targetLayer}_${totalNeuronsCreated}"
        val newNeuron = createNeuron(targetLayer, label, centerX, newY)

        network.addNetworkModelsAsync(newNeuron, usePlacementManager = false)

        // Create synapses
        sourceNeurons.forEach { source ->
            val synapse = Synapse(source, newNeuron, 1.0)
            network.addNetworkModelAsync(synapse)
        }

        neuronsByLayer[targetLayer]?.add(newNeuron)
        predictionRecords[newNeuron] = PredictionRecord(newNeuron)

        println(" Created $label at ($centerX, $newY)")
    }

    fun clearAllNeurons() {
        for (layer in 1..6) {
            neuronsByLayer[layer]?.toList()?.forEach { neuron ->
                runBlocking { neuron.delete() }
            }
            neuronsByLayer[layer]?.clear()
        }
        neuronsByLayer[0]?.forEach { neuron ->
            neuron.activation = 0.0
            neuron.clamped = false
        }
        alreadyCreatedAssociations.clear()
        predictionRecords.clear()
        firingHistory.clear()
        totalNeuronsCreated = 0
        timestep = 0
        astrocytes.forEach { it.calcium = 0.0 }
        astrocyteNodes.forEach { it.paint = Color(255, 220, 50, 0) }
        originalSynapseWeights.forEach { (synapse, original) ->
            synapse.strength = original
        }
        originalSynapseWeights.clear()
        println("Cleared all association neurons, reset base layer and astrocytes")
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // INITIALIZATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Create base layer
    val numBaseNeurons = 12
    val spacing = 60.0
    val baseY = 200.0

    val baseNeuronsList = mutableListOf<Neuron>()
    for (i in 0 until numBaseNeurons) {
        val x = -300.0 + i.toDouble() * spacing
        val neuron = createNeuron(0, "B_$i", x, baseY)
        baseNeuronsList.add(neuron)
        neuronsByLayer[0]!!.add(neuron)
    }

    network.addNetworkModelsAsync(*baseNeuronsList.toTypedArray(), usePlacementManager = false)
    println("Created $numBaseNeurons base neurons in Layer 0 at Y=$baseY")

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // MAIN UPDATE LOOP
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    network.updateManager.clear()
    workspace.addUpdateAction("DAN Update", position = 0) {
        timestep++

        // Record firing across all layers
        val firedByLayer = (0..6).associate { layer ->
            layer to (neuronsByLayer[layer]?.filter { it.activation > firingThreshold }?.toSet() ?: emptySet())
        }.filterValues { it.isNotEmpty() }

        firingHistory.add(TimestepRecord(timestep, firedByLayer))
        if (firingHistory.size > temporalWindow) firingHistory.removeAt(0)

        // Detect co-firing and create associations in all layers
        for (sourceLayer in 0..5) {
            val targetLayer = sourceLayer + 1

            // Build history for the layer
            val layerHistory = firingHistory
                .mapNotNull { it.firedNeuronsByLayer[sourceLayer] }
                .filter { it.size >= 2 }

            if (layerHistory.size >= 2) {
                // Count co-firing patterns
                val coFiringCounts = layerHistory
                    .groupingBy { NeuronGroup(it) }
                    .eachCount()

                // Create new associations
                coFiringCounts
                    .filterValues { it >= coFireThreshold }
                    .filterKeys { it !in alreadyCreatedAssociations }
                    .forEach { (group, count) ->
                        println(" Layer $sourceLayer co-firing: ${group.neurons.map{it.label}} ($count×) → Layer $targetLayer")
                        createAssociationNeuron(group.neurons, targetLayer)
                        alreadyCreatedAssociations.add(group)
                        totalNeuronsCreated++
                    }
            }
        }

        // Expand astrocyte grid if neurons are above existing tiles
        if (astrocytesEnabled) {
            val topY = astrocytes.minOf { it.y }
            val allNeurons = neuronsByLayer.values.flatten()
            val minNeuronY = allNeurons.minOfOrNull { it.y } ?: topY
            if (minNeuronY < topY) {
                var newRowY = topY - astrocyteTileHeight
                while (newRowY > minNeuronY - astrocyteTileHeight) {
                    addAstrocyteRow(newRowY)
                    newRowY -= astrocyteTileHeight
                }
            }
        }

        // Update association neuron activations + track predictions
        for (layer in 1..6) {
            neuronsByLayer[layer]?.forEach { assocNeuron ->
                val inputs = assocNeuron.fanIn
                if (inputs.isNotEmpty()) {
                    val sourceNeurons = inputs.map { it.source }.toSet()
                    val weightedSum = inputs.sumOf { synapse -> synapse.source.activation * synapse.strength } / inputs.size
                    val allActive = weightedSum > firingThreshold

                    assocNeuron.activation = if (weightedSum > firingThreshold) 1.0 else 0.0

                    // Predictive pruning logic
                    if (pruningEnabled) {
                        predictionRecords[assocNeuron]?.let { record ->
                            val activeSources = sourceNeurons.filter { it.activation > firingThreshold }
                            val someButNotAll = activeSources.isNotEmpty() && !allActive

                            if (someButNotAll) {
                                // Make prediction
                                val predicted = sourceNeurons - activeSources.toSet()
                                record.activePredictions.add(Pair(timestep, predicted))
                                record.totalPredictions++
                                println(" ${assocNeuron.label} PREDICTION at t=$timestep: ${activeSources.map{it.label}} fired, predicting ${predicted.map{it.label}} within $pruningWindow steps")
                            }

                            // Check prediction
                            record.activePredictions.removeIf { (predTime, predicted) ->
                                val elapsed = timestep - predTime
                                when {
                                    elapsed <= pruningWindow && predicted.all { it.activation > firingThreshold } -> {
                                        record.correctPredictions++
                                        println(" ${assocNeuron.label} CORRECT prediction. Predicted ${predicted.map{it.label}} and they all fired")
                                        true
                                    }
                                    elapsed > pruningWindow -> {
                                        println(" ${assocNeuron.label} FAILED prediction. Predicted ${predicted.map{it.label}} but window expired")
                                        true
                                    }
                                    else -> false // wait
                                }
                            }
                        }
                    }
                }
            }
        }

// Prune bad predictors
        if (pruningEnabled) {
            // Debug: Show which neurons are being evaluated
            if (timestep % 20 == 0L && predictionRecords.isNotEmpty()) {
                println(" Pruning evaluation at t=$timestep:")
                predictionRecords.forEach { (neuron, record) ->
                    val layer = neuronsByLayer.entries.find { it.value.contains(neuron) }?.key ?: -1
                    val pruningAllowed = layer > 0 && layerPruningEnabled[layer] == true
                    val accuracy = if (record.totalPredictions > 0) {
                        record.correctPredictions.toDouble() / record.totalPredictions * 100
                    } else 0.0
                    val orphaned = neuron.fanIn.isEmpty()
                    val status = when {
                        !pruningAllowed -> "PROTECTED (pruning OFF)"
                        orphaned -> "ORPHANED (no inputs)"
                        record.totalPredictions < pruningThreshold -> "Too few predictions (${record.totalPredictions}/$pruningThreshold)"
                        accuracy < 50.0 -> " WILL PRUNE (${accuracy.toInt()}%)"
                        else -> " Safe (${accuracy.toInt()}%)"
                    }
                    println("   ${neuron.label} (L$layer): $status")
                }
            }

            // Prune neurons with poor prediction accuracy
            val toPrune = predictionRecords.filter { (neuron, record) ->
                val layer = neuronsByLayer.entries.find { it.value.contains(neuron) }?.key ?: -1
                layer > 0 &&
                        layerPruningEnabled[layer] == true &&
                        record.totalPredictions >= pruningThreshold &&
                        record.correctPredictions.toDouble() / record.totalPredictions < 0.5
            }

            toPrune.forEach { (neuron, record) ->
                val accuracy = record.correctPredictions.toDouble() / record.totalPredictions * 100
                println(" Pruning ${neuron.label} (${accuracy.toInt()}% accuracy, ${record.correctPredictions}/${record.totalPredictions})")
                runBlocking { neuron.delete() }
                predictionRecords.remove(neuron)
                neuronsByLayer.values.forEach { it.remove(neuron) }
            }

            // Prune orphaned neurons (neurons with no inputs)
            val orphaned = predictionRecords.filter { (neuron, _) ->
                val layer = neuronsByLayer.entries.find { it.value.contains(neuron) }?.key ?: -1
                layer > 0 &&
                        layerPruningEnabled[layer] == true &&
                        neuron.fanIn.isEmpty()
            }

            orphaned.forEach { (neuron, _) ->
                println(" Pruning ${neuron.label} (orphaned - no inputs)")
                runBlocking { neuron.delete() }
                predictionRecords.remove(neuron)
                neuronsByLayer.values.forEach { it.remove(neuron) }
            }
        }

        // Astrocyte update
        if (astrocytesEnabled) {
            astrocytes.forEachIndexed { i, astro ->
                val allNeurons = neuronsByLayer.values.flatten()
                val neuronsInBounds = allNeurons.filter { neuron ->
                    neuron.x >= astro.x && neuron.x < astro.x + astro.width &&
                    neuron.y >= astro.y && neuron.y < astro.y + astro.height
                }
                // Find synapses whose target neuron is in this astrocyte's bounds
                val synapsesInBounds = neuronsInBounds
                    .flatMap { neuron -> neuron.fanIn }

                // Calcium driven by synaptic transmission in territory
                val synapticActivity = synapsesInBounds.sumOf { synapse: Synapse -> synapse.source.activation * synapse.strength }
                astro.calcium = ((astro.calcium + synapticActivity * astrocyteGain) * astrocyteCalciumDecay).coerceIn(0.0, 1.0)
                if (astro.calcium < 0.01) astro.calcium = 0.0
                astrocyteNodes.getOrNull(i)?.paint = Color(255, 220, 50, (astro.calcium * 80).toInt())

                // Store original weights for new synapses
                synapsesInBounds.forEach { synapse: Synapse ->
                    originalSynapseWeights.putIfAbsent(synapse, synapse.strength)
                }

                // Hard threshold suppression: shut down transmission above threshold
                if (astro.calcium >= astrocyteSuppressionThreshold) {
                    synapsesInBounds.forEach { synapse: Synapse ->
                        synapse.strength = 0.0
                    }
                }

                // Hard threshold recovery: restore original weights below threshold
                if (astro.calcium < astrocyteRecoveryThreshold) {
                    synapsesInBounds.forEach { synapse: Synapse ->
                        val original = originalSynapseWeights[synapse] ?: synapse.strength
                        synapse.strength = original
                    }
                }
            }
        }

        // Apply decay to base layer only [skips clamped neurons]
        val decay = decayRates[0] ?: 0.9
        neuronsByLayer[0]?.forEach { neuron ->
            if (!neuron.clamped) {
                neuron.activation *= decay
            }
        }
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // GUI
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    withGui {
        place(networkComponent) {
            location = point(0, 0)
            width = 800
            height = 600
        }

        // Add astrocyte background tiles
        val networkPanel = getNetworkPanel(networkComponent)
        canvasLayer = networkPanel.canvas.layer
        astrocytes.forEach { astro ->
            val rect = PPath.createRectangle(astro.x, astro.y, astro.width, astro.height)
            rect.paint = Color(255, 220, 50, 0)
            rect.strokePaint = Color(200, 180, 50, 40)
            rect.pickable = false
            networkPanel.canvas.layer.addChild(0, rect)
            astrocyteNodes.add(rect)
        }

        val controlPanel = createControlPanel("DAN Controls", 820, 0) {

            // ~~~~ QUICK ACTIONS ~~~~
            addLabel("<html><b>QUICK ACTIONS</b></html>")

            addButton("Clear All (Reset to Base)") { clearAllNeurons() }

            addButton("Release All") {
                neuronsByLayer[0]?.forEach { it.clamped = false }
            }

            // ~~~~ CORE PARAMETERS ~~~~
            addSeparator()
            addLabel("<html><b>CORE PARAMETERS</b></html>")

            val thresholdField = addTextField("Co-fire threshold (1-20)", coFireThreshold.toString())
            val windowField = addTextField("Temporal window (1-100)", temporalWindow.toString())
            val baseDecayField = addTextField("Base layer decay (0.0-1.0)", decayRates[0].toString())

            // ~~~~ PRUNING ~~~~
            addSeparator()
            addLabel("<html><b>PRUNING</b></html>")

            val pruningEnabledField = addTextField("Enabled (true/false)", pruningEnabled.toString())
            val pruningThresholdField = addTextField("Min predictions (1-100)", pruningThreshold.toString())

            // ~~~~ ASTROCYTE CONTROLS ~~~~
            addSeparator()
            addLabel("<html><b>ASTROCYTE CONTROLS</b></html>")

            val astroEnabledField = addTextField("Enabled (true/false)", astrocytesEnabled.toString())
            val astroDecayField = addTextField("Calcium decay (0.90-0.99)", astrocyteCalciumDecay.toString())
            val astroSuppressionThreshField = addTextField("Suppression threshold (0.0-1.0)", astrocyteSuppressionThreshold.toString())
            val astroRecoveryThreshField = addTextField("Recovery threshold (0.0-1.0)", astrocyteRecoveryThreshold.toString())
            val astroSuppressionStrengthField = addTextField("Suppression strength (0.80-0.99)", astrocyteWeightSuppression.toString())

            addButton("Reset Astrocytes") {
                astrocytes.forEach { it.calcium = 0.0 }
                astrocyteNodes.forEachIndexed { i, node ->
                    node.paint = Color(255, 220, 50, 0)
                }
                originalSynapseWeights.forEach { (synapse, original) ->
                    synapse.strength = original
                }
                originalSynapseWeights.clear()
                println("Astrocytes reset: calcium zeroed, synapse weights restored")
            }

            // ~~~~ APPLY ~~~~
            addSeparator()
            addButton("Apply All Parameters") {
                thresholdField.text.toIntOrNull()?.let { if (it in 1..20) coFireThreshold = it }
                windowField.text.toIntOrNull()?.let { if (it in 1..100) temporalWindow = it }
                baseDecayField.text.toDoubleOrNull()?.let { if (it in 0.0..1.0) decayRates[0] = it }

                pruningEnabledField.text.toBooleanStrictOrNull()?.let { pruningEnabled = it }
                pruningThresholdField.text.toIntOrNull()?.let { if (it in 1..100) pruningThreshold = it }

                astroEnabledField.text.toBooleanStrictOrNull()?.let { astrocytesEnabled = it }
                astroDecayField.text.toDoubleOrNull()?.let { if (it in 0.90..0.99) astrocyteCalciumDecay = it }
                astroSuppressionThreshField.text.toDoubleOrNull()?.let { if (it in 0.0..1.0) astrocyteSuppressionThreshold = it }
                astroRecoveryThreshField.text.toDoubleOrNull()?.let { if (it in 0.0..1.0) astrocyteRecoveryThreshold = it }
                astroSuppressionStrengthField.text.toDoubleOrNull()?.let { if (it in 0.80..0.99) astrocyteWeightSuppression = it }

                println("Parameters applied")
            }

            // ~~~~ STATS ~~~~
            addSeparator()
            addLabel("<html><b>NETWORK STATS</b></html>")
            val statsLabel = addLabel("")

            workspace.addUpdateAction("Update Stats") {
                val counts = (0..6).associate { layer ->
                    layer to (neuronsByLayer[layer]?.size ?: 0)
                }
                val calciumStr = astrocytes.mapIndexed { i, a ->
                    "A$i:${"%.2f".format(a.calcium)}"
                }.joinToString(" ")
                statsLabel.text = "<html>L0:${counts[0]} L1:${counts[1]} L2:${counts[2]} L3:${counts[3]}<br>" +
                        "L4:${counts[4]} L5:${counts[5]} L6:${counts[6]} | t=$timestep<br>" +
                        "Ca: $calciumStr</html>"
            }
        }

        // Make control panel scrollable and size it
        controlPanel.setSize(400, 700)
        val scrollPane = javax.swing.JScrollPane(controlPanel.mainPanel)
        scrollPane.verticalScrollBarPolicy = javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS
        scrollPane.horizontalScrollBarPolicy = javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER
        scrollPane.border = null

        controlPanel.centralPanel.removeAll()
        controlPanel.centralPanel.add(scrollPane, java.awt.BorderLayout.CENTER)
        controlPanel.revalidate()
        controlPanel.repaint()
    }

    // Sidebar documentation
    addSidebarInfo("""
        # Dynamic Associative Network (DAN)

        Hierarchical temporal learning through pattern detection,
        predictive growth, dynamic pruning, and astrocyte modulation.


        ## Architecture

        Seven layers (0-6) form a processing hierarchy:
        - Layer 0: Input neurons (12 base units), activated with the wand tool
        - Layers 1-6: Emergent association neurons

        Each higher layer detects co-firing patterns in the layer below,
        creating progressive hierarchical representations.

        Astrocyte tiles cover the network as a background grid. Each tile
        represents one astrocyte domain that modulates the synapses of
        neurons within its territory.


        ## Core Mechanisms

        **1. Temporal Pattern Detection**
        Neurons that fire together within a temporal window (default: 5 timesteps)
        form associations. When the same group co-fires repeatedly (threshold: 3
        times), a new association neuron is created in the next layer.

        **2. Predictive Growth**
        Association neurons represent predictive relationships. When a subset of
        neurons fires, the association predicts the remaining neurons will
        activate within a prediction window (default: 3 timesteps).

        **3. Dynamic Pruning**
        Associations track prediction accuracy. After accumulating sufficient
        predictions (threshold: 10), neurons with accuracy below 50% are removed.
        Pruning is off by default. Note: pruning and astrocyte modulation can
        conflict, since astrocyte suppression temporarily silences neurons which
        the pruning system may interpret as failed predictions.

        **4. Astrocyte Modulation**
        Astrocyte tiles sense synaptic transmission (source activation times
        synapse weight) flowing into neurons in their territory. This drives a
        slow calcium accumulator that decays gradually (default: 0.97 per step).
        When calcium crosses the suppression threshold, all synapse weights in
        the territory are set to zero, silencing downstream neurons. When calcium
        decays below the recovery threshold, weights are restored to their
        original values. This delayed negative feedback loop produces oscillatory
        dynamics in the association layers: activity builds calcium, calcium
        shuts down transmission, silence lets calcium decay, recovery restores
        transmission, and the cycle repeats. The astrocyte grid expands upward
        automatically as new association neurons are created.


        ## Experimental Workflow

        **Observing Oscillation:**
        1. Set base layer decay to 1.0 so input neurons hold activation
        2. Activate a group of base neurons using the wand tool
        3. Wait for association neurons to form
        4. Watch the astrocyte tile behind the association neurons
        5. The tile color (yellow) shows calcium level building up
        6. When calcium crosses threshold, association neurons go silent
        7. Tile fades as calcium decays
        8. When calcium drops below recovery threshold, neurons fire again
        9. The cycle repeats

        **Comparing With and Without Astrocytes:**
        1. Run the above experiment with astrocytes enabled
        2. Clear All, then disable astrocytes in the control panel
        3. Repeat the same activation pattern
        4. Observe that association neurons stay tonically active with no oscillation


        ## Tunable Parameters

        **Co-fire Threshold (1-20, default: 3)**
        Number of co-occurrences required before creating an association.

        **Temporal Window (1-100, default: 5)**
        Timesteps considered when detecting co-firing patterns.

        **Base Layer Decay (0.0-1.0, default: 1.0)**
        Activation decay rate for input neurons each timestep.
        1.0 means no decay (neurons hold activation).
        Lower values cause neurons to fade after activation.

        **Pruning (off by default)**
        Enable to remove association neurons with poor prediction accuracy.
        Disable when testing astrocyte dynamics.

        **Astrocyte Controls:**
        - Enabled: toggle astrocyte modulation on/off
        - Calcium decay (0.90-0.99): how fast calcium decays each step
        - Suppression threshold (0.0-1.0): calcium level that triggers weight suppression
        - Recovery threshold (0.0-1.0): calcium level that triggers weight restoration
        - Reset Astrocytes: zeros calcium and restores all synapse weights
    """.trimIndent())
}