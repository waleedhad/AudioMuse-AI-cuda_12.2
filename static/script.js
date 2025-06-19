// --- DOM Element References ---
const loadingSpinner = document.getElementById('loading-spinner');
const mainContent = document.getElementById('main-content');

// View switcher
const basicViewBtn = document.getElementById('basic-view-btn');
const advancedViewBtn = document.getElementById('advanced-view-btn');
const advancedParams = document.querySelectorAll('.advanced-param');
const kmeansParamsBasic = document.getElementById('kmeans-params-basic');
const basicAlgorithmDisplay = document.getElementById('basic-algorithm-display');


// Config Form
const clusterAlgorithmSelect = document.getElementById('config-cluster_algorithm');
const dbscanParamsDiv = document.getElementById('dbscan-params');
// const kmeansParamsDiv = document.getElementById('kmeans-params'); // This element doesn't exist, kmeansParamsBasic is used
const gmmParamsDiv = document.getElementById('gmm-params');
const aiModelProviderSelect = document.getElementById('config-ai_model_provider');
const ollamaConfigGroup = document.getElementById('ollama-config-group');
const geminiConfigGroup = document.getElementById('gemini-config-group');

// Task Buttons
const startAnalysisBtn = document.getElementById('start-analysis-btn');
const startClusteringBtn = document.getElementById('start-clustering-btn');
const fetchPlaylistsBtn = document.getElementById('fetch-playlists-btn');
const cancelTaskBtn = document.getElementById('cancel-task-btn');

// Task Status Display
const statusTaskId = document.getElementById('status-task-id');
const statusTaskType = document.getElementById('status-task-type');
const statusStatus = document.getElementById('status-status');
const statusProgress = document.getElementById('status-progress');
const progressBar = document.getElementById('progress-bar');
const statusDetails = document.getElementById('status-details');

// Playlists
const playlistsSection = document.getElementById('playlists-section');
const playlistsContainer = document.getElementById('playlists-container');

// --- State Variables ---
let currentTaskId = null;
let lastPolledTaskDetails = {};

// --- Functions ---

/**
 * Switches between basic and advanced configuration views.
 * @param {('basic'|'advanced')} viewToShow The view to display.
 */
function switchView(viewToShow) {
    if (viewToShow === 'basic') {
        basicViewBtn.classList.add('active');
        advancedViewBtn.classList.remove('active');
        advancedParams.forEach(el => el.classList.add('hidden'));
        if (kmeansParamsBasic) kmeansParamsBasic.classList.remove('hidden'); // Show K-Means params in basic
        if (basicAlgorithmDisplay) basicAlgorithmDisplay.classList.remove('hidden');
        if (clusterAlgorithmSelect) clusterAlgorithmSelect.classList.add('hidden'); // Hide algorithm dropdown in basic

        // In basic view, we only support K-Means
        if (clusterAlgorithmSelect) {
            clusterAlgorithmSelect.value = 'kmeans';
        }

    } else { // advanced view
        basicViewBtn.classList.remove('active');
        advancedViewBtn.classList.add('active');
        advancedParams.forEach(el => el.classList.remove('hidden'));
        kmeansParamsBasic.classList.add('hidden'); // Hide the basic K-Means inputs
        basicAlgorithmDisplay.classList.add('hidden');
        if (clusterAlgorithmSelect) clusterAlgorithmSelect.classList.remove('hidden'); // Show algorithm dropdown in advanced
    }
     // This function handles showing/hiding algorithm-specific params
    toggleClusteringParams();
}

function showLoading(show) {
    loadingSpinner.style.display = show ? 'flex' : 'none';
    mainContent.style.display = show ? 'none' : 'block';
}

async function fetchConfig() {
    showLoading(true);
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        renderConfig(config);
        // Call switchView here to ensure the view is set correctly *before* showing the content
        switchView('basic'); 
        toggleAiConfig();
    } catch (error) {
        console.error('Error fetching config:', error);
        showMessageBox('Error', 'Failed to load configuration. Please check the backend server.');
    } finally {
        showLoading(false);
    }
}

function renderConfig(config) {
    // General
    document.getElementById('config-jellyfin_url').value = config.jellyfin_url || '';
    document.getElementById('config-jellyfin_user_id').value = config.jellyfin_user_id || '';
    document.getElementById('config-jellyfin_token').value = config.jellyfin_token || '';

    // Analysis
    document.getElementById('config-num_recent_albums').value = config.num_recent_albums || 0;
    document.getElementById('config-top_n_moods').value = config.top_n_moods || 0;

    // Clustering
    clusterAlgorithmSelect.value = (config.cluster_algorithm === 'dbscan' || config.cluster_algorithm === 'gmm') ? config.cluster_algorithm : 'kmeans';
    document.getElementById('config-max_distance').value = config.max_distance || 0;
    document.getElementById('config-max_songs_per_cluster').value = config.max_songs_per_cluster || 0;
    document.getElementById('config-pca_components_min').value = config.pca_components_min || 0;
    document.getElementById('config-pca_components_max').value = config.pca_components_max || 0;
    document.getElementById('config-clustering_runs').value = config.clustering_runs || 0;
    document.getElementById('config-min_songs_per_genre_for_stratification').value = config.min_songs_per_genre_for_stratification || 0;
    document.getElementById('config-stratified_sampling_target_percentile').value = config.stratified_sampling_target_percentile || 0;
    document.getElementById('config-score_weight_diversity').value = config.score_weight_diversity || 0;
    document.getElementById('config-score_weight_purity').value = config.score_weight_purity || 0;
    document.getElementById('config-score_weight_silhouette').value = config.score_weight_silhouette || 0;
    document.getElementById('config-score_weight_davies_bouldin').value = config.score_weight_davies_bouldin || 0;
    document.getElementById('config-score_weight_calinski_harabasz').value = config.score_weight_calinski_harabasz || 0;
    document.getElementById('config-score_weight_other_feature_diversity').value = config.score_weight_other_feature_diversity || 0;
    document.getElementById('config-score_weight_other_feature_purity').value = config.score_weight_other_feature_purity || 0;

    // Algorithm Specific
    document.getElementById('config-dbscan_eps_min').value = config.dbscan_eps_min || 0;
    document.getElementById('config-dbscan_eps_max').value = config.dbscan_eps_max || 0;
    document.getElementById('config-dbscan_min_samples_min').value = config.dbscan_min_samples_min || 0;
    document.getElementById('config-dbscan_min_samples_max').value = config.dbscan_min_samples_max || 0;
    document.getElementById('config-num_clusters_min').value = config.num_clusters_min || 0;
    document.getElementById('config-num_clusters_max').value = config.num_clusters_max || 0;
    document.getElementById('config-gmm_n_components_min').value = config.gmm_n_components_min || 0;
    document.getElementById('config-gmm_n_components_max').value = config.gmm_n_components_max || 0;

    // AI Naming
    aiModelProviderSelect.value = config.ai_model_provider || 'NONE';
    document.getElementById('config-ollama_server_url').value = config.ollama_server_url || 'http://127.0.0.1:11434/api/generate';
    document.getElementById('config-ollama_model_name').value = config.ollama_model_name || 'mistral:7b';
    document.getElementById('config-gemini_api_key').value = config.gemini_api_key || '';
    document.getElementById('config-gemini_model_name').value = config.gemini_model_name || 'gemini-1.5-flash-latest';
}

function toggleClusteringParams() {
    const selectedAlgorithm = clusterAlgorithmSelect.value;
    dbscanParamsDiv.classList.add('hidden');
    gmmParamsDiv.classList.add('hidden');
    // K-Means params (kmeansParamsBasic) are handled based on view below

    // Only show algorithm-specific params in advanced view
    if (advancedViewBtn.classList.contains('active')) {
        // First, ensure K-Means params are hidden if K-Means is NOT selected
        if (selectedAlgorithm !== 'kmeans' && kmeansParamsBasic) {
            kmeansParamsBasic.classList.add('hidden');
        }

        if (selectedAlgorithm === 'dbscan') {
            dbscanParamsDiv.classList.remove('hidden');
        } else if (selectedAlgorithm === 'gmm') {
            gmmParamsDiv.classList.remove('hidden');
        } else if (selectedAlgorithm === 'kmeans' && kmeansParamsBasic) { 
            kmeansParamsBasic.classList.remove('hidden'); // Show K-Means params if K-Means is selected
        }
    }
}

function toggleAiConfig() {
    const provider = aiModelProviderSelect.value;
    ollamaConfigGroup.classList.add('hidden');
    geminiConfigGroup.classList.add('hidden');

    if (provider === 'OLLAMA') {
        ollamaConfigGroup.classList.remove('hidden');
    } else if (provider === 'GEMINI') {
        geminiConfigGroup.classList.remove('hidden');
    }
}

function updateCancelButtonState(isDisabled) {
    cancelTaskBtn.disabled = isDisabled;
    cancelTaskBtn.style.opacity = isDisabled ? '0.5' : '1';
    cancelTaskBtn.style.cursor = isDisabled ? 'not-allowed' : 'pointer';
}

async function checkActiveTasks() {
    try {
        const response = await fetch('/api/active_tasks');
        const mainActiveTask = await response.json(); 

        if (mainActiveTask && mainActiveTask.task_id) {
            currentTaskId = mainActiveTask.task_id;
            const currentStatusUpper = (mainActiveTask.status || mainActiveTask.state || 'UNKNOWN').toUpperCase();
            const terminalStates = ['SUCCESS', 'FINISHED', 'FAILURE', 'FAILED', 'REVOKED', 'CANCELED'];

            displayTaskStatus(mainActiveTask); 
            
            const previousStateWasTerminal = lastPolledTaskDetails[currentTaskId]?.state && terminalStates.includes(lastPolledTaskDetails[currentTaskId].state.toUpperCase());

            if (terminalStates.includes(currentStatusUpper) && !previousStateWasTerminal) {
                let alertTitle = 'Task Update';
                let alertMessage = `Task ${mainActiveTask.task_id} (${mainActiveTask.task_type_from_db || 'Unknown Type'}) has ${currentStatusUpper.toLowerCase()}.`;
                if (['SUCCESS', 'FINISHED'].includes(currentStatusUpper)) alertTitle = 'Task Completed';
                else if (['FAILURE', 'FAILED'].includes(currentStatusUpper)) alertTitle = 'Task Failed';
                else if (['REVOKED', 'CANCELED'].includes(currentStatusUpper)) alertTitle = 'Task Canceled';
                
                showMessageBox(alertTitle, alertMessage);
            }
            lastPolledTaskDetails[currentTaskId] = { state: currentStatusUpper, ...mainActiveTask };
            disableTaskButtons(true);
            updateCancelButtonState(false);
            return true; 
        } else if (currentTaskId) {
            const finishedTaskId = currentTaskId;
            const previousDetails = lastPolledTaskDetails[finishedTaskId];
            currentTaskId = null;

            try {
                const finalStatusResponse = await fetch(`/api/status/${finishedTaskId}`);
                if (finalStatusResponse.ok) {
                    const finalStatusData = await finalStatusResponse.json();
                    const upperFinalStatus = (finalStatusData.state || 'UNKNOWN').toUpperCase();
                    const terminalStates = ['SUCCESS', 'FINISHED', 'FAILURE', 'FAILED', 'REVOKED', 'CANCELED'];
                    
                    const finalStatusIsTerminal = terminalStates.includes(upperFinalStatus);
                    const previousStateWasTerminal = previousDetails && previousDetails.state && terminalStates.includes(previousDetails.state.toUpperCase());

                    if (finalStatusIsTerminal && !previousStateWasTerminal) {
                        let alertTitle = 'Task Update';
                        let alertMessage = `Task ${finalStatusData.task_id} (${finalStatusData.task_type_from_db || 'Unknown Type'}) has ${upperFinalStatus.toLowerCase()}.`;
                        if (['SUCCESS', 'FINISHED'].includes(upperFinalStatus)) alertTitle = 'Task Completed';
                        else if (['FAILURE', 'FAILED'].includes(upperFinalStatus)) alertTitle = 'Task Failed';
                        else if (['REVOKED', 'CANCELED'].includes(upperFinalStatus)) alertTitle = 'Task Canceled';
                        
                        showMessageBox(alertTitle, alertMessage);
                    }
                    displayTaskStatus(finalStatusData);
                } else {
                    showMessageBox('Task Finished', `Task ${finishedTaskId} is no longer active. Final status could not be retrieved.`);
                    displayTaskStatus({ task_id: finishedTaskId, state: 'FINISHED', progress: 100 });
                }
            } catch (e) {
                 console.error(`Error fetching final status for ${finishedTaskId}:`, e);
            } finally {
                delete lastPolledTaskDetails[finishedTaskId];
                disableTaskButtons(false);
                updateCancelButtonState(true);
            }
            return true;
        } else {
            disableTaskButtons(false);
            updateCancelButtonState(true);
        }
    } catch (error) {
        console.error('Error checking active tasks:', error);
        displayTaskStatus({ task_id: 'Error', task_type: 'Error', state: 'ERROR', progress: 0, details: `Polling error: ${error.message}` });
        currentTaskId = null;
        disableTaskButtons(false); 
        updateCancelButtonState(true);
    }
    return false;
}

function disableTaskButtons(isDisabled) {
    startAnalysisBtn.disabled = isDisabled;
    startClusteringBtn.disabled = isDisabled;
    fetchPlaylistsBtn.disabled = isDisabled;
    
    [startAnalysisBtn, startClusteringBtn, fetchPlaylistsBtn].forEach(btn => {
        btn.style.opacity = isDisabled ? '0.5' : '1';
        btn.style.cursor = isDisabled ? 'not-allowed' : 'pointer';
    });
}

function displayTaskStatus(task) {
    statusTaskId.textContent = task.task_id || 'N/A';
    statusTaskType.textContent = task.task_type_from_db || task.task_type || 'N/A';
    const stateUpper = (task.state || task.status || 'IDLE').toUpperCase();
    statusStatus.textContent = stateUpper;
    statusProgress.textContent = task.progress || 0;
    progressBar.style.width = `${task.progress || 0}%`;

    let color = '#fbbf24'; // Yellow for pending/progress
    if (['SUCCESS', 'FINISHED'].includes(stateUpper)) color = '#4ade80'; // Green for success
    else if (['FAILURE', 'FAILED', 'REVOKED', 'CANCELED'].includes(stateUpper)) color = '#f87171'; // Red for failure
    else if (stateUpper === 'IDLE') color = '#d1d5db'; // Gray for idle

    statusStatus.style.color = color;
    
    if (['SUCCESS', 'FINISHED'].includes(stateUpper) && (task.task_type_from_db || task.task_type || '').toLowerCase().includes('clustering')) {
        fetchPlaylists(); 
    }

    statusDetails.textContent = typeof task.details === 'object' ? JSON.stringify(task.details, null, 2) : task.details;
    statusDetails.scrollTop = statusDetails.scrollHeight;
}

async function startTask(taskType) {
    disableTaskButtons(true);
    updateCancelButtonState(true);

    const payload = {
        jellyfin_url: document.getElementById('config-jellyfin_url').value,
        jellyfin_user_id: document.getElementById('config-jellyfin_user_id').value,
        jellyfin_token: document.getElementById('config-jellyfin_token').value
    };

    if (taskType === 'analysis') {
        payload.num_recent_albums = parseInt(document.getElementById('config-num_recent_albums').value);
        payload.top_n_moods = parseInt(document.getElementById('config-top_n_moods').value);
    } else if (taskType === 'clustering') {
        playlistsSection.style.display = 'none';
        playlistsContainer.innerHTML = '';
        
        // Always collect all params, backend can decide what to use based on algorithm
        Object.assign(payload, {
            clustering_method: clusterAlgorithmSelect.value,
            max_distance: parseFloat(document.getElementById('config-max_distance').value),
            max_songs_per_cluster: parseInt(document.getElementById('config-max_songs_per_cluster').value),
            pca_components_min: parseInt(document.getElementById('config-pca_components_min').value),
            pca_components_max: parseInt(document.getElementById('config-pca_components_max').value),
            clustering_runs: parseInt(document.getElementById('config-clustering_runs').value),
            min_songs_per_genre_for_stratification: parseInt(document.getElementById('config-min_songs_per_genre_for_stratification').value),
            stratified_sampling_target_percentile: parseInt(document.getElementById('config-stratified_sampling_target_percentile').value),
            score_weight_diversity: parseFloat(document.getElementById('config-score_weight_diversity').value),
            score_weight_purity: parseFloat(document.getElementById('config-score_weight_purity').value),
            score_weight_silhouette: parseFloat(document.getElementById('config-score_weight_silhouette').value),
            score_weight_davies_bouldin: parseFloat(document.getElementById('config-score_weight_davies_bouldin').value),
            score_weight_calinski_harabasz: parseFloat(document.getElementById('config-score_weight_calinski_harabasz').value),
            score_weight_other_feature_diversity: parseFloat(document.getElementById('config-score_weight_other_feature_diversity').value),
            score_weight_other_feature_purity: parseFloat(document.getElementById('config-score_weight_other_feature_purity').value),
            dbscan_eps_min: parseFloat(document.getElementById('config-dbscan_eps_min').value),
            dbscan_eps_max: parseFloat(document.getElementById('config-dbscan_eps_max').value),
            dbscan_min_samples_min: parseInt(document.getElementById('config-dbscan_min_samples_min').value),
            dbscan_min_samples_max: parseInt(document.getElementById('config-dbscan_min_samples_max').value),
            num_clusters_min: parseInt(document.getElementById('config-num_clusters_min').value),
            num_clusters_max: parseInt(document.getElementById('config-num_clusters_max').value),
            gmm_n_components_min: parseInt(document.getElementById('config-gmm_n_components_min').value),
            gmm_n_components_max: parseInt(document.getElementById('config-gmm_n_components_max').value),
            ai_model_provider: aiModelProviderSelect.value,
            ollama_server_url: document.getElementById('config-ollama_server_url').value,
            ollama_model_name: document.getElementById('config-ollama_model_name').value,
            gemini_api_key: document.getElementById('config-gemini_api_key').value,
            gemini_model_name: document.getElementById('config-gemini_model_name').value,
            enable_clustering_embeddings: document.getElementById('config-enable_clustering_embeddings').checked // Added this line
        });
    }

    try {
        const response = await fetch(`/api/${taskType}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        if (response.ok && result.task_id) {
            currentTaskId = result.task_id;
            displayTaskStatus({ task_id: result.task_id, task_type: result.task_type, state: 'PENDING', progress: 0, details: 'Task enqueued.' });
            lastPolledTaskDetails[result.task_id] = { state: 'PENDING', task_type: result.task_type, task_id: result.task_id };
            updateCancelButtonState(false);
        } else {
            throw new Error(result.message || 'Failed to start task.');
        }
    } catch (error) {
        console.error(`Error starting ${taskType} task:`, error);
        showMessageBox('Error', `Failed to start ${taskType} task: ${error.message}`);
        disableTaskButtons(false);
        updateCancelButtonState(true);
    }
}

async function cancelTask() {
    if (!currentTaskId) return;
    updateCancelButtonState(true);
    try {
        const response = await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' });
        const result = await response.json();
        if (response.ok) {
            showMessageBox('Success', result.message);
            checkActiveTasks();
        } else {
            throw new Error(result.message || 'Failed to cancel task.');
        }
    } catch (error) {
        console.error('Error cancelling task:', error);
        showMessageBox('Error', `Failed to cancel task: ${error.message}`);
        updateCancelButtonState(false);
    }
}

async function fetchPlaylists() {
    playlistsContainer.innerHTML = '<p style="color: #9ca3af;">Fetching playlists...</p>';
    playlistsSection.style.display = 'block';
    try {
        const response = await fetch('/api/playlists');
        if (!response.ok) throw new Error(`Server responded with ${response.status}`);
        const playlistsData = await response.json();
        renderPlaylists(playlistsData);
    } catch (error) {
        console.error('Error fetching playlists:', error);
        playlistsContainer.innerHTML = `<p style="color: #f87171;">Error fetching playlists: ${error.message}</p>`;
    }
}

function renderPlaylists(playlistsData) {
    playlistsContainer.innerHTML = '';
    if (!playlistsData || Object.keys(playlistsData).length === 0) {
        playlistsContainer.innerHTML = '<p style="color: #9ca3af;">No playlists found.</p>';
        return;
    }
    for (const [playlistName, songs] of Object.entries(playlistsData)) {
        const playlistDiv = document.createElement('div');
        playlistDiv.style.cssText = 'margin-bottom: 1rem; padding: 1rem; background-color: #1f2937; border-radius: 0.375rem;';
        playlistDiv.innerHTML = `
            <p style="color: #93c5fd;">
                <strong style="color: #e5e7eb;">${playlistName}</strong> - (${songs.length} songs)
                <button class="show-songs-btn" style="margin-left: 0.5rem; background-color: #3b82f6; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 0.25rem; cursor: pointer;">SHOW</button>
            </p>
            <ul class="song-list" style="display: none; margin-top: 0.5rem; list-style-type: disc; padding-left: 1.5rem;">
                ${songs.map(song => `<li>${song.title} by ${song.author}</li>`).join('')}
            </ul>`;
        playlistDiv.querySelector('.show-songs-btn').addEventListener('click', e => {
            const btn = e.target;
            const list = btn.closest('div').querySelector('.song-list');
            list.style.display = list.style.display === 'none' ? 'block' : 'none';
            btn.textContent = list.style.display === 'none' ? 'SHOW' : 'HIDE';
        });
        playlistsContainer.appendChild(playlistDiv);
    }
}

function showMessageBox(title, message) {
    const boxId = 'custom-message-box';
    document.getElementById(boxId)?.remove();
    const messageBox = document.createElement('div');
    messageBox.id = boxId;
    messageBox.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: #333; color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index: 1000; border: 1px solid #3b82f6; max-width: 400px; text-align: center;';
    messageBox.innerHTML = `<h3 style="font-weight: bold; margin-bottom: 10px; color: #93c5fd;">${title}</h3><p>${message}</p><button style="margin-top: 15px; padding: 8px 15px; background-color: #3b82f6; color: white; border: none; border-radius: 5px; cursor: pointer;" onclick="this.parentNode.remove()">OK</button>`;
    document.body.appendChild(messageBox);
}

async function fetchAndDisplayOverallLastTask() {
    try {
        const response = await fetch('/api/last_task');
        if (response.ok) {
            const lastTask = await response.json();
            if (lastTask && lastTask.task_id) displayTaskStatus(lastTask);
            else displayTaskStatus({ state: 'IDLE', details: 'No previous task found.' });
        } else {
            displayTaskStatus({ state: 'IDLE', details: 'Could not fetch last task status.' });
        }
    } catch (error) {
        console.error('Error fetching last task status:', error);
        displayTaskStatus({ state: 'IDLE', details: 'Error fetching last task status.' });
    }
}

// --- Event Listeners & Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    await fetchConfig();
    if (!await checkActiveTasks()) {
        await fetchAndDisplayOverallLastTask();
        updateCancelButtonState(true);
    }
    setInterval(checkActiveTasks, 3000);
});

basicViewBtn.addEventListener('click', () => switchView('basic'));
advancedViewBtn.addEventListener('click', () => switchView('advanced'));
clusterAlgorithmSelect.addEventListener('change', toggleClusteringParams);
aiModelProviderSelect.addEventListener('change', toggleAiConfig);
startAnalysisBtn.addEventListener('click', () => startTask('analysis'));
startClusteringBtn.addEventListener('click', () => startTask('clustering'));
fetchPlaylistsBtn.addEventListener('click', fetchPlaylists);
cancelTaskBtn.addEventListener('click', cancelTask);
