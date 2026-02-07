import React, { useState } from 'react';
import axios from 'axios';
import {
    AppBar,
    Toolbar,
    Typography,
    Container,
    Grid,
    Paper,
    Box,
    TextField,
    Button,
    CircularProgress,
    Alert,
    Chip,
    ThemeProvider,
    createTheme,
    CssBaseline,
    Tabs,
    Tab,
    IconButton,
    Tooltip,
    List,
    ListItem,
    ListItemButton,
    ListItemText,
    Divider,
    Fade
} from '@mui/material';
import {
    CloudUpload as CloudUploadIcon,
    Settings as SettingsIcon,
    PlayArrow as PlayArrowIcon,
    CheckCircle as CheckCircleIcon,
    Psychology as PsychologyIcon,
    Description as DescriptionIcon,
    BubbleChart as BubbleChartIcon,
    Input as InputIcon,
    Info as InfoIcon,
    HelpOutline as HelpIcon
} from '@mui/icons-material';

// Black & White High Contrast Theme
const bwTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: '#ffffff' },
        secondary: { main: '#9e9e9e' },
        background: { default: '#000000', paper: '#0a0a0a' },
        text: { primary: '#ffffff', secondary: '#b0b0b0' },
        divider: '#333333',
    },
    typography: {
        fontFamily: 'Roboto, sans-serif',
        h4: { fontWeight: 700, letterSpacing: '-0.02em' },
        h5: { fontWeight: 600 },
        h6: { fontWeight: 600 },
        button: { fontWeight: 700 },
    },
    components: {
        MuiPaper: {
            styleOverrides: {
                root: { backgroundImage: 'none', border: '1px solid #333' },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 0,
                    border: '1px solid transparent',
                    '&:hover': { border: '1px solid #fff' }
                },
                contained: {
                    backgroundColor: '#fff',
                    color: '#000',
                    '&:hover': { backgroundColor: '#e0e0e0' },
                },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: '#333',
                    color: '#fff',
                    fontSize: '0.875rem',
                    border: '1px solid #555'
                }
            }
        }
    },
});

function App() {
    const [file, setFile] = useState(null);
    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [config, setConfig] = useState({ numClusters: 5, provider: 'gemini' });

    // Top-level Tabs: 0=Input, 1=Summary, 2=Clustering
    const [mainTab, setMainTab] = useState(0);
    // Sub-selection for Clustering Tab
    const [selectedClusterIndex, setSelectedClusterIndex] = useState(0);

    const handleFileChange = (e) => {
        if (e.target.files[0]) {
            setFile(e.target.files[0]);
            setText("");
        }
    };

    const handleSummarize = async () => {
        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        if (file) {
            formData.append('file', file);
        } else if (text) {
            formData.append('text', text);
        } else {
            setError("Please provide text or upload a file.");
            setLoading(false);
            return;
        }

        try {
            const response = await axios.post('/api/summarize', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResult(response.data);
            setMainTab(1); // Auto-switch to Summary tab on success
        } catch (err) {
            setError(err.response?.data?.error || err.message);
        } finally {
            setLoading(false);
        }
    };

    // --- Render Functions for Each Tab ---

    const renderInputTab = () => (
        <Container maxWidth="lg" sx={{ py: 4, height: '100%', overflowY: 'auto' }}>
            <Grid container spacing={6} alignItems="center" justifyContent="center" sx={{ minHeight: '80vh' }}>
                <Grid item xs={12} md={6}>
                    <Box sx={{ mb: 4 }}>
                        <Typography variant="h3" gutterBottom fontWeight="bold">
                            HERCULES
                        </Typography>
                        <Typography variant="h6" color="text.secondary" gutterBottom>
                            Hierarchical Embedding-based Recursive Clustering
                        </Typography>
                        <Typography variant="body1" color="text.secondary" sx={{ mt: 2, maxWidth: 500 }}>
                            Upload your large documents or paste text to generate structured, hierarchical summaries.
                            We use advanced semantic clustering to break down complex topics.
                        </Typography>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                        <Tooltip title="Start here by uploading a file or pasting text" arrow placement="top">
                            <Chip icon={<InfoIcon />} label="Step 1: Input Data" variant="outlined" />
                        </Tooltip>
                        <Tooltip title="Adjust how many topic clusters you want" arrow placement="top">
                            <Chip icon={<InfoIcon />} label="Step 2: Configure" variant="outlined" />
                        </Tooltip>
                        <Tooltip title="Run the AI pipeline" arrow placement="top">
                            <Chip icon={<InfoIcon />} label="Step 3: Analyze" variant="outlined" />
                        </Tooltip>
                    </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                    <Paper elevation={0} sx={{ p: 4, bgcolor: '#050505' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <CloudUploadIcon sx={{ mr: 1 }} />
                            <Typography variant="h6">Data Source</Typography>
                            <Box sx={{ flexGrow: 1 }} />
                            <Tooltip title="Supported formats: .txt (UTF-8). PDF support coming soon.">
                                <HelpIcon fontSize="small" color="secondary" />
                            </Tooltip>
                        </Box>

                        <Box
                            sx={{
                                border: '1px dashed #444',
                                p: 4,
                                textAlign: 'center',
                                cursor: 'pointer',
                                position: 'relative',
                                mb: 3,
                                '&:hover': { borderColor: '#fff', bgcolor: '#111' }
                            }}
                        >
                            <input
                                type="file"
                                onChange={handleFileChange}
                                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0, cursor: 'pointer' }}
                            />
                            <Typography variant="body1" fontWeight="bold">
                                {file ? file.name : "Click to Upload File"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                                Drag & drop or browse
                            </Typography>
                        </Box>

                        <Divider sx={{ my: 3 }}>OR</Divider>

                        <TextField
                            fullWidth
                            multiline
                            rows={4}
                            placeholder="Paste your text content here..."
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            variant="outlined"
                            sx={{ mb: 3 }}
                        />

                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <SettingsIcon sx={{ mr: 1 }} />
                            <Typography variant="h6">Configuration</Typography>
                            <Box sx={{ flexGrow: 1 }} />
                            <Tooltip title="Determines how many distinct topic groups the AI will attempt to find in your text.">
                                <HelpIcon fontSize="small" color="secondary" />
                            </Tooltip>
                        </Box>

                        <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
                            <TextField
                                label="Target Clusters"
                                type="number"
                                fullWidth
                                value={config.numClusters}
                                onChange={(e) => setConfig({ ...config, numClusters: parseInt(e.target.value) })}
                                InputProps={{ inputProps: { min: 1, max: 20 } }}
                                helperText="Recommended: 3-10"
                            />
                        </Box>

                        <Button
                            variant="contained"
                            fullWidth
                            size="large"
                            onClick={handleSummarize}
                            disabled={loading}
                            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
                            sx={{ py: 2 }}
                        >
                            {loading ? 'RUNNING PIPELINE...' : 'START ANALYSIS'}
                        </Button>

                        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );

    const renderSummaryTab = () => (
        <Container maxWidth="lg" sx={{ py: 4, height: '100%', overflowY: 'auto' }}>
            <Box sx={{ maxWidth: 900, mx: 'auto' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 4, justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <DescriptionIcon fontSize="large" />
                        <Typography variant="h4">Executive Summary</Typography>
                    </Box>
                    <Tooltip title="This is a high-level synthesis of all identified clusters.">
                        <HelpIcon color="secondary" />
                    </Tooltip>
                </Box>

                <Paper sx={{ p: 6, bgcolor: '#000', border: '1px solid #333', lineHeight: 1.8 }}>
                    <Typography variant="body1" sx={{ fontSize: '1.15rem', whiteSpace: 'pre-wrap', fontFamily: 'Georgia, serif' }}>
                        {result?.final_summary}
                    </Typography>
                </Paper>

                <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
                    <Chip label={`Analysis Time: ${result?.duration_seconds}s`} variant="outlined" />
                </Box>
            </Box>
        </Container>
    );

    const renderClusteringTab = () => (
        <Box sx={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
            {/* Sidebar List */}
            <Box sx={{ width: 300, borderRight: '1px solid #333', overflowY: 'auto', bgcolor: '#050505' }}>
                <Box sx={{ p: 2, borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="subtitle1" fontWeight="bold">Identified Clusters</Typography>
                    <Tooltip title="These are the distinct semantic groups found in your text.">
                        <HelpIcon fontSize="small" color="secondary" />
                    </Tooltip>
                </Box>
                <List>
                    {result?.cluster_summaries.map((_, idx) => (
                        <ListItemButton
                            key={idx}
                            selected={selectedClusterIndex === idx}
                            onClick={() => setSelectedClusterIndex(idx)}
                            sx={{
                                borderLeft: selectedClusterIndex === idx ? '3px solid #fff' : '3px solid transparent',
                                bgcolor: selectedClusterIndex === idx ? '#111' : 'transparent'
                            }}
                        >
                            <BubbleChartIcon sx={{ mr: 2, fontSize: 20, color: selectedClusterIndex === idx ? '#fff' : '#666' }} />
                            <ListItemText
                                primary={`Cluster ${idx + 1}`}
                                secondary={`Topic Group ${idx + 1}`}
                                primaryTypographyProps={{ fontWeight: selectedClusterIndex === idx ? 'bold' : 'normal' }}
                            />
                        </ListItemButton>
                    ))}
                </List>
            </Box>

            {/* Main Content */}
            <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 6 }}>
                <Box sx={{ maxWidth: 800, mx: 'auto' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 4, gap: 2 }}>
                        <Typography variant="h4">Cluster {selectedClusterIndex + 1} Analysis</Typography>
                        <Tooltip title="Detailed summary of this specific topic cluster.">
                            <HelpIcon color="secondary" />
                        </Tooltip>
                    </Box>

                    <Paper sx={{ p: 5, bgcolor: '#000', border: '1px solid #333' }}>
                        <Typography variant="body1" sx={{ fontSize: '1.1rem', whiteSpace: 'pre-wrap', lineHeight: 1.8, fontFamily: 'Georgia, serif' }}>
                            {result?.cluster_summaries[selectedClusterIndex].replace(/^Cluster \d+:\n/, '')}
                        </Typography>
                    </Paper>
                </Box>
            </Box>
        </Box>
    );

    return (
        <ThemeProvider theme={bwTheme}>
            <CssBaseline />
            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: 'background.default' }}>

                {/* Main Navigation Bar */}
                <AppBar position="static" color="default" elevation={0} sx={{ bgcolor: '#000', borderBottom: '1px solid #333' }}>
                    <Toolbar>
                        <PsychologyIcon sx={{ mr: 2, fontSize: 28 }} />
                        <Typography variant="h6" sx={{ mr: 4, fontWeight: 900, letterSpacing: '.1rem' }}>HERCULES</Typography>

                        <Tabs
                            value={mainTab}
                            onChange={(e, v) => setMainTab(v)}
                            textColor="primary"
                            indicatorColor="primary"
                            sx={{ flexGrow: 1 }}
                        >
                            <Tab icon={<InputIcon />} iconPosition="start" label="Input & Config" />
                            <Tab icon={<DescriptionIcon />} iconPosition="start" label="Summary" disabled={!result} />
                            <Tab icon={<BubbleChartIcon />} iconPosition="start" label="Clustering Info" disabled={!result} />
                        </Tabs>

                        {result && (
                            <Tooltip title="Reset and start a new analysis">
                                <Button color="inherit" onClick={() => { setResult(null); setMainTab(0); }}>
                                    New Analysis
                                </Button>
                            </Tooltip>
                        )}
                    </Toolbar>
                </AppBar>

                {/* Content Area */}
                <Box sx={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
                    {mainTab === 0 && renderInputTab()}
                    {mainTab === 1 && renderSummaryTab()}
                    {mainTab === 2 && renderClusteringTab()}
                </Box>

            </Box>
        </ThemeProvider>
    );
}

export default App;
