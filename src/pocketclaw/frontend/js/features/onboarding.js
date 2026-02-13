/**
 * PocketPaw - Onboarding Feature Module
 *
 * Created: 2026-02-13
 *
 * Handles first-time user experience:
 * - Multi-step setup wizard
 * - Getting started checklist
 * - Contextual tours and highlights
 * - Sample prompts and quick-start presets
 */

window.PocketPaw = window.PocketPaw || {};

window.PocketPaw.Onboarding = {
    name: 'Onboarding',

    /**
     * Get initial state for Onboarding
     */
    getState() {
        return {
            // Wizard state
            setupStep: 1, // 1: API Key, 2: Test, 3: Quick Start
            setupTotalSteps: 3,
            isTestingConnection: false,
            testMessageSent: false,
            testResponseReceived: false,
            selectedPreset: null,

            // Checklist state
            checklistMinimized: false,
            checklistDismissed: localStorage.getItem('pocketpaw_checklist_dismissed') === '1',
            checklistItems: {
                apiKey: { label: 'Add API Key', complete: false },
                firstMessage: { label: 'Send first message', complete: false },
                useSkill: { label: 'Try a skill', complete: false },
                setupChannel: { label: 'Connect a channel', complete: false }
            },

            // Preset configurations
            presets: [
                {
                    id: 'developer',
                    icon: 'code',
                    title: 'Developer',
                    description: 'Write and debug code, analyze projects',
                    systemPrompt: 'You are a helpful coding assistant. You write clean, well-documented code and explain your reasoning.',
                    samplePrompts: [
                        'Write a Python script to fetch data from an API',
                        'Debug this code and explain what\'s wrong',
                        'Help me set up a Docker container for my project'
                    ]
                },
                {
                    id: 'productivity',
                    icon: 'zap',
                    title: 'Productivity',
                    description: 'Tasks, scheduling, and organization',
                    systemPrompt: 'You are a productivity assistant. Help with task management, scheduling, and staying organized.',
                    samplePrompts: [
                        'Create a todo list for my week',
                        'Help me plan my day efficiently',
                        'Summarize this document for me'
                    ]
                },
                {
                    id: 'system',
                    icon: 'terminal',
                    title: 'System Admin',
                    description: 'Manage files, system tasks, automation',
                    systemPrompt: 'You are a system administration assistant. Help with file management, shell commands, and automation.',
                    samplePrompts: [
                        'Show me disk usage and large files',
                        'Create a backup script for my documents',
                        'Help me configure my development environment'
                    ]
                }
            ],

            // Sample prompts for empty state
            samplePrompts: [
                { text: 'Write a Python script to...', icon: 'code' },
                { text: 'Analyze this code for bugs', icon: 'search' },
                { text: 'Search for latest AI news', icon: 'globe' },
                { text: 'Set a reminder for...', icon: 'bell' },
                { text: 'Create a todo list', icon: 'check-square' },
                { text: 'Help me plan my day', icon: 'calendar' }
            ],

            // Tour state
            tourActive: false,
            tourStep: 0,
            tourSteps: [
                { target: '.messages', title: 'Chat Area', description: 'Your conversations appear here. Try sending a message!' },
                { target: 'input[type="text"]', title: 'Message Input', description: 'Type your questions or commands here.' },
                { target: '.settings-nav-item', title: 'Settings', description: 'Configure API keys, behavior, and channels here.' },
                { target: '.sidebar', title: 'Sidebar', description: 'Access your chat history, projects, and tools.' }
            ]
        };
    },

    /**
     * Get methods for Onboarding
     */
    getMethods() {
        return {
            /**
             * Initialize onboarding state from localStorage
             */
            initOnboarding() {
                // Load checklist completion status
                const savedChecklist = localStorage.getItem('pocketpaw_checklist');
                if (savedChecklist) {
                    try {
                        const parsed = JSON.parse(savedChecklist);
                        this.checklistItems = { ...this.checklistItems, ...parsed };
                    } catch (e) {
                        console.warn('[Onboarding] Failed to load checklist:', e);
                    }
                }

                // Update API key status immediately
                this.checklistItems.apiKey.complete = this.hasAnthropicKey;

                // Check if wizard should show (no API key + not previously completed)
                const onboardingComplete = localStorage.getItem('pocketpaw_onboarding_complete') === '1';
                if (!this.hasAnthropicKey && !onboardingComplete) {
                    this.showWelcome = true;
                    this.setupStep = 1;
                }
            },

            /**
             * Advance to next wizard step
             */
            nextSetupStep() {
                if (this.setupStep < this.setupTotalSteps) {
                    this.setupStep++;
                    
                    // If advancing to step 2, trigger test
                    if (this.setupStep === 2) {
                        this.runConnectionTest();
                    }
                } else {
                    this.completeOnboarding();
                }
            },

            /**
             * Go back to previous wizard step
             */
            prevSetupStep() {
                if (this.setupStep > 1) {
                    this.setupStep--;
                }
            },

            /**
             * Run connection test on step 2
             */
            async runConnectionTest() {
                this.isTestingConnection = true;
                this.testMessageSent = false;
                this.testResponseReceived = false;

                // Wait a moment for UI to update
                await new Promise(resolve => setTimeout(resolve, 500));

                // Send test message
                this.testMessageSent = true;
                this.inputText = 'Hello! Can you tell me what you can help me with?';
                this.sendMessage();

                // Wait for response (poll for messages)
                let attempts = 0;
                const checkForResponse = setInterval(() => {
                    attempts++;
                    
                    // Check if we got a response
                    const hasResponse = this.messages.length > 0 && 
                        this.messages.some(m => m.role === 'assistant' && !m.isNew);
                    
                    if (hasResponse || attempts > 30) { // 15 second timeout
                        clearInterval(checkForResponse);
                        this.testResponseReceived = true;
                        this.isTestingConnection = false;
                        
                        // Mark first message checklist item complete
                        this.checklistItems.firstMessage.complete = true;
                        this.saveChecklist();
                    }
                }, 500);
            },

            /**
             * Select a quick-start preset
             */
            selectPreset(presetId) {
                const preset = this.presets.find(p => p.id === presetId);
                if (!preset) return;

                this.selectedPreset = presetId;

                // Set system prompt via memory
                if (socket.isConnected) {
                    socket.send('set_system_prompt', { 
                        prompt: preset.systemPrompt,
                        persist: true 
                    });
                }

                // Inject sample prompts into empty state
                this.samplePrompts = preset.samplePrompts.map(text => ({
                    text,
                    icon: this.getIconForPrompt(text)
                }));

                // Log selection
                this.log(`Selected preset: ${preset.title}`, 'info');
            },

            /**
             * Get appropriate icon for a prompt text
             */
            getIconForPrompt(text) {
                const lower = text.toLowerCase();
                if (lower.includes('code') || lower.includes('python') || lower.includes('script')) return 'code';
                if (lower.includes('search') || lower.includes('find')) return 'search';
                if (lower.includes('remind')) return 'bell';
                if (lower.includes('todo') || lower.includes('list')) return 'check-square';
                if (lower.includes('plan') || lower.includes('schedule')) return 'calendar';
                if (lower.includes('debug') || lower.includes('fix')) return 'bug';
                return 'message-circle';
            },

            /**
             * Apply selected preset and close wizard
             */
            applyPresetAndClose() {
                if (this.selectedPreset) {
                    const preset = this.presets.find(p => p.id === this.selectedPreset);
                    this.showToast(`Ready to help with ${preset.title.toLowerCase()}!`, 'success');
                }
                this.completeOnboarding();
            },

            /**
             * Skip presets and close wizard
             */
            skipPresets() {
                this.completeOnboarding();
            },

            /**
             * Complete the onboarding process
             */
            completeOnboarding() {
                this.showWelcome = false;
                localStorage.setItem('pocketpaw_onboarding_complete', '1');
                localStorage.setItem('pocketpaw_setup_dismissed', '1');
                
                // Mark API key complete
                this.checklistItems.apiKey.complete = true;
                this.saveChecklist();

                // Show checklist
                this.checklistDismissed = false;

                // Log completion
                this.log('Onboarding complete! Welcome to PocketPaw.', 'success');
            },

            /**
             * Save checklist progress to localStorage
             */
            saveChecklist() {
                localStorage.setItem('pocketpaw_checklist', JSON.stringify(this.checklistItems));
            },

            /**
             * Mark a checklist item as complete
             */
            completeChecklistItem(itemKey) {
                if (this.checklistItems[itemKey]) {
                    this.checklistItems[itemKey].complete = true;
                    this.saveChecklist();
                }
            },

            /**
             * Dismiss the checklist permanently
             */
            dismissChecklist() {
                this.checklistDismissed = true;
                localStorage.setItem('pocketpaw_checklist_dismissed', '1');
            },

            /**
             * Check if all checklist items are complete
             */
            get isChecklistComplete() {
                return Object.values(this.checklistItems).every(item => item.complete);
            },

            /**
             * Get completion percentage
             */
            get checklistProgress() {
                const items = Object.values(this.checklistItems);
                const complete = items.filter(i => i.complete).length;
                return Math.round((complete / items.length) * 100);
            },

            /**
             * Start the UI tour
             */
            startTour() {
                this.tourActive = true;
                this.tourStep = 0;
            },

            /**
             * Next tour step
             */
            nextTourStep() {
                if (this.tourStep < this.tourSteps.length - 1) {
                    this.tourStep++;
                } else {
                    this.endTour();
                }
            },

            /**
             * End the tour
             */
            endTour() {
                this.tourActive = false;
                this.tourStep = 0;
            },

            /**
             * Use a sample prompt (click from empty state)
             */
            useSamplePrompt(promptText) {
                this.inputText = promptText;
                // Focus the input
                this.$nextTick(() => {
                    const input = document.querySelector('input[type="text"]');
                    if (input) input.focus();
                });
            },

            /**
             * Check and update checklist based on app events
             */
            updateChecklistFromEvent(eventType, data) {
                switch (eventType) {
                    case 'message_sent':
                        if (!this.checklistItems.firstMessage.complete) {
                            this.completeChecklistItem('firstMessage');
                        }
                        break;
                    case 'skill_used':
                        if (!this.checklistItems.useSkill.complete) {
                            this.completeChecklistItem('useSkill');
                        }
                        break;
                    case 'channel_connected':
                        if (!this.checklistItems.setupChannel.complete) {
                            this.completeChecklistItem('setupChannel');
                        }
                        break;
                    case 'api_key_saved':
                        if (!this.checklistItems.apiKey.complete) {
                            this.completeChecklistItem('apiKey');
                        }
                        break;
                }
            }
        };
    }
};

window.PocketPaw.Loader.register('Onboarding', window.PocketPaw.Onboarding);
