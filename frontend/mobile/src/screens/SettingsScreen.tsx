/**
 * Settings Screen
 * 
 * App settings, security, network configuration, and preferences
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Switch,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

// Theme
const theme = {
  colors: {
    primary: '#6366f1',
    background: '#0f172a',
    card: '#1e293b',
    cardAlt: '#334155',
    text: '#f8fafc',
    textSecondary: '#94a3b8',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    border: '#475569',
  },
};

// Setting Item Component
const SettingItem: React.FC<{
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  subtitle?: string;
  value?: string;
  onPress?: () => void;
  trailing?: React.ReactNode;
  danger?: boolean;
}> = ({ icon, title, subtitle, value, onPress, trailing, danger }) => (
  <TouchableOpacity
    style={styles.settingItem}
    onPress={onPress}
    disabled={!onPress}
  >
    <View style={[styles.settingIcon, danger && styles.settingIconDanger]}>
      <Ionicons
        name={icon}
        size={20}
        color={danger ? theme.colors.error : theme.colors.primary}
      />
    </View>
    <View style={styles.settingContent}>
      <Text style={[styles.settingTitle, danger && styles.settingTitleDanger]}>
        {title}
      </Text>
      {subtitle && <Text style={styles.settingSubtitle}>{subtitle}</Text>}
    </View>
    {value && <Text style={styles.settingValue}>{value}</Text>}
    {trailing}
    {onPress && !trailing && (
      <Ionicons name="chevron-forward" size={20} color={theme.colors.textSecondary} />
    )}
  </TouchableOpacity>
);

// Section Header
const SectionHeader: React.FC<{ title: string }> = ({ title }) => (
  <Text style={styles.sectionHeader}>{title}</Text>
);

export default function SettingsScreen() {
  const [biometricsEnabled, setBiometricsEnabled] = useState(true);
  const [autoLock, setAutoLock] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [haptics, setHaptics] = useState(true);
  const [network, setNetwork] = useState('mainnet');

  const handleNetworkChange = () => {
    Alert.alert(
      'Select Network',
      'Choose the network to connect to',
      [
        {
          text: 'Mainnet',
          onPress: () => setNetwork('mainnet'),
        },
        {
          text: 'Testnet (Sepolia)',
          onPress: () => setNetwork('testnet'),
        },
        {
          text: 'Local (Development)',
          onPress: () => setNetwork('local'),
        },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  const handleExportSeed = () => {
    Alert.alert(
      'Export Recovery Phrase',
      'Your recovery phrase gives full access to your wallet. Never share it with anyone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'I Understand',
          style: 'destructive',
          onPress: () => {
            // Would require biometric auth here
            Alert.alert('Recovery Phrase', 'word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12');
          },
        },
      ]
    );
  };

  const handleClearData = () => {
    Alert.alert(
      'Clear All Data',
      'This will remove all wallets and settings. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear Data',
          style: 'destructive',
          onPress: () => {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            Alert.alert('Data Cleared', 'All data has been removed.');
          },
        },
      ]
    );
  };

  const handleToggle = (setter: React.Dispatch<React.SetStateAction<boolean>>, value: boolean) => {
    Haptics.selectionAsync();
    setter(!value);
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView}>
        {/* Security Section */}
        <SectionHeader title="Security" />
        <View style={styles.section}>
          <SettingItem
            icon="finger-print"
            title="Biometric Authentication"
            subtitle="Use Face ID or fingerprint to unlock"
            trailing={
              <Switch
                value={biometricsEnabled}
                onValueChange={() => handleToggle(setBiometricsEnabled, biometricsEnabled)}
                trackColor={{ false: theme.colors.cardAlt, true: theme.colors.primary }}
              />
            }
          />
          <SettingItem
            icon="lock-closed"
            title="Auto-Lock"
            subtitle="Lock app when backgrounded"
            trailing={
              <Switch
                value={autoLock}
                onValueChange={() => handleToggle(setAutoLock, autoLock)}
                trackColor={{ false: theme.colors.cardAlt, true: theme.colors.primary }}
              />
            }
          />
          <SettingItem
            icon="key"
            title="Export Recovery Phrase"
            subtitle="Backup your wallet"
            onPress={handleExportSeed}
          />
          <SettingItem
            icon="shield-checkmark"
            title="Change PIN"
            onPress={() => Alert.alert('Coming Soon', 'PIN change will be available in the next update.')}
          />
        </View>

        {/* Network Section */}
        <SectionHeader title="Network" />
        <View style={styles.section}>
          <SettingItem
            icon="globe"
            title="Network"
            value={network.charAt(0).toUpperCase() + network.slice(1)}
            onPress={handleNetworkChange}
          />
          <SettingItem
            icon="server"
            title="RPC Endpoint"
            subtitle="Configure custom RPC"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="analytics"
            title="Gas Settings"
            subtitle="Adjust transaction speed"
            onPress={() => Alert.alert('Coming Soon')}
          />
        </View>

        {/* Privacy Section */}
        <SectionHeader title="Privacy" />
        <View style={styles.section}>
          <SettingItem
            icon="eye-off"
            title="Default Privacy Level"
            value="Private (3)"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="time"
            title="Proof Caching"
            subtitle="Cache ZK proofs for faster transactions"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="people"
            title="Anonymity Sets"
            subtitle="Manage mixing pools"
            onPress={() => Alert.alert('Coming Soon')}
          />
        </View>

        {/* Preferences Section */}
        <SectionHeader title="Preferences" />
        <View style={styles.section}>
          <SettingItem
            icon="notifications"
            title="Push Notifications"
            trailing={
              <Switch
                value={notifications}
                onValueChange={() => handleToggle(setNotifications, notifications)}
                trackColor={{ false: theme.colors.cardAlt, true: theme.colors.primary }}
              />
            }
          />
          <SettingItem
            icon="pulse"
            title="Haptic Feedback"
            trailing={
              <Switch
                value={haptics}
                onValueChange={() => handleToggle(setHaptics, haptics)}
                trackColor={{ false: theme.colors.cardAlt, true: theme.colors.primary }}
              />
            }
          />
          <SettingItem
            icon="language"
            title="Language"
            value="English"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="cash"
            title="Currency"
            value="USD"
            onPress={() => Alert.alert('Coming Soon')}
          />
        </View>

        {/* About Section */}
        <SectionHeader title="About" />
        <View style={styles.section}>
          <SettingItem
            icon="information-circle"
            title="About NexusZero"
            subtitle="Version 0.1.0"
            onPress={() => Alert.alert('NexusZero Protocol', 'Privacy-preserving transactions powered by quantum-resistant zero-knowledge proofs.')}
          />
          <SettingItem
            icon="document-text"
            title="Terms of Service"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="shield"
            title="Privacy Policy"
            onPress={() => Alert.alert('Coming Soon')}
          />
          <SettingItem
            icon="logo-github"
            title="GitHub"
            subtitle="github.com/nexuszero"
            onPress={() => Alert.alert('Coming Soon')}
          />
        </View>

        {/* Danger Zone */}
        <SectionHeader title="Danger Zone" />
        <View style={styles.section}>
          <SettingItem
            icon="trash"
            title="Clear All Data"
            subtitle="Remove all wallets and settings"
            onPress={handleClearData}
            danger
          />
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>NexusZero Protocol v0.1.0</Text>
          <Text style={styles.footerSubtext}>© 2024 NexusZero Team</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  scrollView: {
    flex: 1,
  },
  sectionHeader: {
    color: theme.colors.textSecondary,
    fontSize: 14,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginHorizontal: 16,
    marginTop: 24,
    marginBottom: 8,
  },
  section: {
    backgroundColor: theme.colors.card,
    marginHorizontal: 16,
    borderRadius: 12,
    overflow: 'hidden',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.background,
  },
  settingIcon: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: theme.colors.primary + '15',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  settingIconDanger: {
    backgroundColor: theme.colors.error + '15',
  },
  settingContent: {
    flex: 1,
  },
  settingTitle: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '500',
  },
  settingTitleDanger: {
    color: theme.colors.error,
  },
  settingSubtitle: {
    color: theme.colors.textSecondary,
    fontSize: 13,
    marginTop: 2,
  },
  settingValue: {
    color: theme.colors.textSecondary,
    fontSize: 16,
    marginRight: 8,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  footerText: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
  footerSubtext: {
    color: theme.colors.cardAlt,
    fontSize: 12,
    marginTop: 4,
  },
});
