/**
 * Wallet Screen
 * 
 * Manage wallets, view balances, and connect accounts
 */

import React, { useState } from 'react';
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Modal,
  TextInput,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Clipboard from 'expo-clipboard';
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

// Token type
interface Token {
  symbol: string;
  name: string;
  balance: string;
  value: string;
  change: number;
  icon: keyof typeof Ionicons.glyphMap;
}

// Sample tokens
const tokens: Token[] = [
  { symbol: 'ETH', name: 'Ethereum', balance: '2.5', value: '$4,850.00', change: 3.24, icon: 'logo-ethereum' },
  { symbol: 'USDC', name: 'USD Coin', balance: '5,000', value: '$5,000.00', change: 0.01, icon: 'cash-outline' },
  { symbol: 'WBTC', name: 'Wrapped BTC', balance: '0.15', value: '$6,450.00', change: 2.15, icon: 'logo-bitcoin' },
  { symbol: 'NXZ', name: 'NexusZero', balance: '10,000', value: '$1,500.00', change: 12.5, icon: 'shield-checkmark' },
];

// Token Row Component
const TokenRow: React.FC<{ token: Token }> = ({ token }) => (
  <TouchableOpacity style={styles.tokenRow}>
    <View style={styles.tokenIconContainer}>
      <Ionicons name={token.icon} size={28} color={theme.colors.primary} />
    </View>
    <View style={styles.tokenDetails}>
      <Text style={styles.tokenSymbol}>{token.symbol}</Text>
      <Text style={styles.tokenName}>{token.name}</Text>
    </View>
    <View style={styles.tokenBalance}>
      <Text style={styles.tokenValue}>{token.value}</Text>
      <Text style={styles.tokenAmount}>{token.balance} {token.symbol}</Text>
    </View>
    <View style={styles.tokenChange}>
      <Ionicons
        name={token.change >= 0 ? 'trending-up' : 'trending-down'}
        size={16}
        color={token.change >= 0 ? theme.colors.success : theme.colors.error}
      />
      <Text
        style={[
          styles.tokenChangeText,
          { color: token.change >= 0 ? theme.colors.success : theme.colors.error },
        ]}
      >
        {token.change >= 0 ? '+' : ''}{token.change}%
      </Text>
    </View>
  </TouchableOpacity>
);

// Wallet Card Component
const WalletCard: React.FC<{ address: string; name: string; isActive: boolean }> = ({
  address,
  name,
  isActive,
}) => {
  const handleCopyAddress = async () => {
    await Clipboard.setStringAsync(address);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    Alert.alert('Copied', 'Address copied to clipboard');
  };

  return (
    <TouchableOpacity style={[styles.walletCard, isActive && styles.walletCardActive]}>
      <View style={styles.walletHeader}>
        <View style={styles.walletInfo}>
          <Text style={styles.walletName}>{name}</Text>
          <TouchableOpacity onPress={handleCopyAddress}>
            <Text style={styles.walletAddress}>
              {address.slice(0, 6)}...{address.slice(-4)}
              <Ionicons name="copy-outline" size={14} color={theme.colors.textSecondary} />
            </Text>
          </TouchableOpacity>
        </View>
        {isActive && (
          <View style={styles.activeBadge}>
            <Text style={styles.activeBadgeText}>Active</Text>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
};

export default function WalletScreen() {
  const [showAddWallet, setShowAddWallet] = useState(false);
  const [importMethod, setImportMethod] = useState<'mnemonic' | 'privateKey' | null>(null);
  const [inputValue, setInputValue] = useState('');

  const handleCreateWallet = () => {
    Alert.alert(
      'Create New Wallet',
      'A new wallet with a secure mnemonic phrase will be generated.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Create',
          onPress: () => {
            // Would generate wallet here
            Alert.alert('Success', 'New wallet created! Please backup your mnemonic phrase.');
          },
        },
      ]
    );
  };

  const handleImportWallet = () => {
    if (!inputValue.trim()) {
      Alert.alert('Error', 'Please enter a valid mnemonic or private key');
      return;
    }
    // Would import wallet here
    setShowAddWallet(false);
    setImportMethod(null);
    setInputValue('');
    Alert.alert('Success', 'Wallet imported successfully!');
  };

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      <ScrollView style={styles.scrollView}>
        {/* Wallets Section */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Your Wallets</Text>
            <TouchableOpacity onPress={() => setShowAddWallet(true)}>
              <Ionicons name="add-circle-outline" size={24} color={theme.colors.primary} />
            </TouchableOpacity>
          </View>

          <WalletCard
            address="0x1234567890abcdef1234567890abcdef12345678"
            name="Main Wallet"
            isActive={true}
          />
          <WalletCard
            address="0xabcdef1234567890abcdef1234567890abcdef12"
            name="Trading Wallet"
            isActive={false}
          />
        </View>

        {/* Tokens Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Tokens</Text>
          {tokens.map((token) => (
            <TokenRow key={token.symbol} token={token} />
          ))}
        </View>

        {/* Actions */}
        <View style={styles.section}>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="add-outline" size={20} color={theme.colors.text} />
            <Text style={styles.actionButtonText}>Add Custom Token</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="refresh-outline" size={20} color={theme.colors.text} />
            <Text style={styles.actionButtonText}>Refresh Balances</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Add Wallet Modal */}
      <Modal
        visible={showAddWallet}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowAddWallet(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Add Wallet</Text>
              <TouchableOpacity onPress={() => {
                setShowAddWallet(false);
                setImportMethod(null);
                setInputValue('');
              }}>
                <Ionicons name="close" size={24} color={theme.colors.text} />
              </TouchableOpacity>
            </View>

            {!importMethod ? (
              <View style={styles.modalOptions}>
                <TouchableOpacity style={styles.modalOption} onPress={handleCreateWallet}>
                  <Ionicons name="add-circle" size={32} color={theme.colors.primary} />
                  <Text style={styles.modalOptionTitle}>Create New Wallet</Text>
                  <Text style={styles.modalOptionDesc}>Generate a new wallet with secure mnemonic</Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={styles.modalOption}
                  onPress={() => setImportMethod('mnemonic')}
                >
                  <Ionicons name="document-text" size={32} color={theme.colors.success} />
                  <Text style={styles.modalOptionTitle}>Import with Mnemonic</Text>
                  <Text style={styles.modalOptionDesc}>12 or 24 word recovery phrase</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.modalOption}
                  onPress={() => setImportMethod('privateKey')}
                >
                  <Ionicons name="key" size={32} color={theme.colors.warning} />
                  <Text style={styles.modalOptionTitle}>Import with Private Key</Text>
                  <Text style={styles.modalOptionDesc}>Hexadecimal private key</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.importForm}>
                <Text style={styles.importLabel}>
                  {importMethod === 'mnemonic' ? 'Enter Mnemonic Phrase' : 'Enter Private Key'}
                </Text>
                <TextInput
                  style={styles.importInput}
                  placeholder={
                    importMethod === 'mnemonic'
                      ? 'word1 word2 word3 ...'
                      : '0x...'
                  }
                  placeholderTextColor={theme.colors.textSecondary}
                  value={inputValue}
                  onChangeText={setInputValue}
                  multiline={importMethod === 'mnemonic'}
                  secureTextEntry={importMethod === 'privateKey'}
                  autoCapitalize="none"
                  autoCorrect={false}
                />
                <TouchableOpacity style={styles.importButton} onPress={handleImportWallet}>
                  <Text style={styles.importButtonText}>Import Wallet</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.backButton}
                  onPress={() => setImportMethod(null)}
                >
                  <Text style={styles.backButtonText}>Back</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        </View>
      </Modal>
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
  section: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    color: theme.colors.text,
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },
  walletCard: {
    backgroundColor: theme.colors.card,
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  walletCardActive: {
    borderColor: theme.colors.primary,
  },
  walletHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  walletInfo: {},
  walletName: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  walletAddress: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
  activeBadge: {
    backgroundColor: theme.colors.primary + '20',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  activeBadgeText: {
    color: theme.colors.primary,
    fontSize: 12,
    fontWeight: '600',
  },
  tokenRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  tokenIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: theme.colors.cardAlt,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  tokenDetails: {
    flex: 1,
  },
  tokenSymbol: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
  },
  tokenName: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
  tokenBalance: {
    alignItems: 'flex-end',
    marginRight: 12,
  },
  tokenValue: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
  },
  tokenAmount: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    marginTop: 2,
  },
  tokenChange: {
    flexDirection: 'row',
    alignItems: 'center',
    width: 60,
  },
  tokenChangeText: {
    fontSize: 12,
    fontWeight: '500',
    marginLeft: 4,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  actionButtonText: {
    color: theme.colors.text,
    fontSize: 16,
    marginLeft: 12,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: theme.colors.background,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  modalTitle: {
    color: theme.colors.text,
    fontSize: 24,
    fontWeight: 'bold',
  },
  modalOptions: {},
  modalOption: {
    backgroundColor: theme.colors.card,
    borderRadius: 16,
    padding: 20,
    marginBottom: 12,
  },
  modalOptionTitle: {
    color: theme.colors.text,
    fontSize: 18,
    fontWeight: '600',
    marginTop: 12,
  },
  modalOptionDesc: {
    color: theme.colors.textSecondary,
    fontSize: 14,
    marginTop: 4,
  },
  importForm: {
    paddingTop: 8,
  },
  importLabel: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  importInput: {
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 16,
    color: theme.colors.text,
    fontSize: 16,
    minHeight: 100,
    textAlignVertical: 'top',
    marginBottom: 16,
  },
  importButton: {
    backgroundColor: theme.colors.primary,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 12,
  },
  importButtonText: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: '600',
  },
  backButton: {
    alignItems: 'center',
    padding: 12,
  },
  backButtonText: {
    color: theme.colors.textSecondary,
    fontSize: 16,
  },
});
